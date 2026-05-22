/*
 * Auteur      : Projet RESCAL-ALS
 * Date        : 2026
 * Description : allocations, conversions CSR et opérations matricielles.
 */

#include "utiles.h"

int g_num_threads = 1; /* valeur par defaut : sequentiel */

typedef struct {
    /* Matrices d'entree (lecture seule, partagees) */
    double** A_data;    /* donnees de A */
    double** B_data;    /* donnees de B */
    double** C_data;    /* donnees de C (ecriture, zone exclusive) */

    /* Dimensions */
    int M;              /* lignes de A = lignes de C */
    int K;              /* colonnes de A = lignes de B */
    int N;              /* colonnes de B = colonnes de C */

    /* Plage de lignes assignee a CE thread */
    int row_start;      /* premiere ligne (incluse) */
    int row_end;        /* derniere ligne (exclue)  */

} DotThreadArgs;

typedef struct {
    CSRMatrix* A;
    Matrix* B;
    Matrix* C;
    int row_start;
    int row_end;
} DotCsrThreadArgs;

typedef struct {
    Matrix* A;
    CSRMatrix* B;
    Matrix* C;
    int row_start;
    int row_end;
} DotCsr1ThreadArgs;

static void* thread_block_dot(void* arg) {
    DotThreadArgs* args = (DotThreadArgs*)arg;

    double** A = args->A_data;
    double** B = args->B_data;
    double** C = args->C_data;
    int K = args->K;
    int N = args->N;
    int row_start = args->row_start;
    int row_end   = args->row_end;

    /* Cache blocking : boucles sur les blocs */
    /* Chaque bloc de taille BLOCK_SIZE tient dans le cache L1 */
    for (int i = row_start; i < row_end; i += BLOCK_SIZE) {
        for (int k = 0; k < K; k += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {

                /* Bornes des blocs (gestion des bords) */
                int i_end = i + BLOCK_SIZE < row_end ? i + BLOCK_SIZE : row_end;
                int k_end = k + BLOCK_SIZE < K      ? k + BLOCK_SIZE : K;
                int j_end = j + BLOCK_SIZE < N      ? j + BLOCK_SIZE : N;

                /* Micro-kernel : calcul sur le bloc */
                /* Ordre (ii, kk, jj) pour localite spatiale optimale */
                /* A[ii][kk] charge une fois, reutilise pour tout j   */
                /* Compilateur vectorise la boucle jj avec -march=native */
                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        double a_val = A[ii][kk]; /* charge 1 fois */
                        for (int jj = j; jj < j_end; jj++) {
                            C[ii][jj] += a_val * B[kk][jj];
                        }
                    }
                }

            }
        }
    }

    return NULL;
}

static void distribute_rows(int rows, int p, int t, int* row_start, int* row_end) {
    int base_rows = rows / p;
    int remainder = rows % p;
    int start = t * base_rows + (t < remainder ? t : remainder);
    int rows_for_t = base_rows + (t < remainder ? 1 : 0);

    *row_start = start;
    *row_end = start + rows_for_t;
}

static void* thread_dot_csr(void* arg) {
    DotCsrThreadArgs* args = (DotCsrThreadArgs*)arg;
    CSRMatrix* A = args->A;
    Matrix* B = args->B;
    Matrix* C = args->C;
    const int p = B->cols;

    for (int i = args->row_start; i < args->row_end; i++) {
        double* C_row = C->data[i];
        const int row_start = A->row_ptr[i];
        const int row_end = A->row_ptr[i + 1];

        for (int k = row_start; k < row_end; k++) {
            const int col = A->col_index[k];
            const double val = A->values[k];
            double* B_row = B->data[col];

            for (int j = 0; j < p; j++) {
                C_row[j] += val * B_row[j];
            }
        }
    }

    return NULL;
}

static void* thread_dot_csr1(void* arg) {
    DotCsr1ThreadArgs* args = (DotCsr1ThreadArgs*)arg;
    Matrix* A = args->A;
    CSRMatrix* B = args->B;
    Matrix* C = args->C;

    /* Strategie choisie : decoupe row-wise sur les lignes de A.
       B reste au format CSR ; chaque thread accumule exclusivement
       dans ses lignes de C, ce qui evite toute synchronisation. */
    for (int i = args->row_start; i < args->row_end; i++) {
        double* A_row = A->data[i];
        double* C_row = C->data[i];

        for (int k = 0; k < B->rows; k++) {
            const int row_start = B->row_ptr[k];
            const int row_end = B->row_ptr[k + 1];
            const double a_val = A_row[k];

            for (int idx = row_start; idx < row_end; idx++) {
                const int j = B->col_index[idx];
                const double val = B->values[idx];
                C_row[j] += a_val * val;
            }
        }
    }

    return NULL;
}

void check_slices(Tensor3D* X) {
    if (X == NULL) {
        fprintf(stderr, "Erreur : tenseur NULL\n");
        exit(EXIT_FAILURE);
    }
    if (X->num_slices <= 0) {
        fprintf(stderr, "Erreur : tenseur sans tranche\n");
        exit(EXIT_FAILURE);
    }
    
    int rows = X->slices[0].rows;
    int cols = X->slices[0].cols;

    for (int i = 0; i < X->num_slices; i++){  
        if (X->slices[i].rows <= 0 || X->slices[i].cols <= 0){
            fprintf(stderr, "Erreur : dimensions de tranche invalides\n");
            exit(EXIT_FAILURE);
        }
        else if (X->slices[i].rows != rows || X->slices[i].cols != cols) {
            fprintf(stderr, "Erreur : dimensions de tranches incoherentes\n");
            exit(EXIT_FAILURE);
        }
    }
}

double norm(Matrix* matrix) {
    if (matrix == NULL || matrix->data == NULL || matrix->rows <= 0 || matrix->cols <= 0) {
        fprintf(stderr, "Erreur : matrice invalide\n");
        return -1.0;
    }

    double norm = 0.0;
    for (int i = 0; i < matrix->rows; i += BLOCK_SIZE) {
        for (int j = 0; j < matrix->cols; j += BLOCK_SIZE) {
            int i_end = i + BLOCK_SIZE < matrix->rows ? i + BLOCK_SIZE : matrix->rows;
            int j_end = j + BLOCK_SIZE < matrix->cols ? j + BLOCK_SIZE : matrix->cols;
            for (int ii = i; ii < i_end; ii++) {
                if (matrix->data[ii] == NULL) {
                    fprintf(stderr, "Erreur : ligne %d invalide\n", ii);
                    return -1.0;
                }
                for (int jj = j; jj < j_end; jj++) {
                    double val = matrix->data[ii][jj];
                    norm += val * val;
                }
            }
        }
    }
    return sqrt(norm);
}

int count_non_zero(const Matrix* mat) {
    if (mat == NULL || mat->data == NULL || mat->rows <= 0 || mat->cols <= 0) {
        fprintf(stderr, "Erreur : matrice invalide\n");
        return -1;
    }

    int count = 0;
    for (int i = 0; i < mat->rows; i += BLOCK_SIZE) {
        for (int j = 0; j < mat->cols; j += BLOCK_SIZE) {
            int i_end = i + BLOCK_SIZE < mat->rows ? i + BLOCK_SIZE : mat->rows;
            int j_end = j + BLOCK_SIZE < mat->cols ? j + BLOCK_SIZE : mat->cols;
            for (int ii = i; ii < i_end; ++ii) {
                if (mat->data[ii] == NULL) {
                    fprintf(stderr, "Erreur : ligne %d invalide\n", ii);
                    return -1;
                }
                for (int jj = j; jj < j_end; ++jj) {
                    if (mat->data[ii][jj] != 0.0) {
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

CSRMatrix* dense_to_csr(Matrix* dense) {
    if (!dense || !dense->data || dense->rows <= 0 || dense->cols <= 0) {
        fprintf(stderr, "Erreur : Matrice dense invalide\n");
        return NULL;
    }

    size_t num_rows = (size_t)dense->rows;
    size_t num_cols = (size_t)dense->cols;

    /* Premier parcours : comptage par ligne. */
    int* nnz_per_row = (int*)calloc(num_rows, sizeof(int));
    if (!nnz_per_row) {
        fprintf(stderr, "Erreur : Allocation de nnz_per_row échouée\n");
        return NULL;
    }

    for (size_t i = 0; i < num_rows; i += BLOCK_SIZE) {
        size_t i_end = i + BLOCK_SIZE < num_rows ? i + BLOCK_SIZE : num_rows;
        for (size_t j = 0; j < num_cols; j += BLOCK_SIZE) {
            size_t j_end = j + BLOCK_SIZE < num_cols ? j + BLOCK_SIZE : num_cols;
            for (size_t ii = i; ii < i_end; ++ii) {
                if (!dense->data[ii]) {
                    fprintf(stderr, "Erreur : Ligne %zu invalide\n", ii);
                    free(nnz_per_row);
                    return NULL;
                }
                for (size_t jj = j; jj < j_end; ++jj) {
                    if (fabs(dense->data[ii][jj]) > EPSILON) {
                        nnz_per_row[ii]++;
                    }
                }
            }
        }
    }

    int nnz = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        nnz += nnz_per_row[i];
    }

    CSRMatrix* csr = init_CSRMatrix(dense->rows, dense->cols, nnz);
    if (!csr) {
        fprintf(stderr, "Erreur : Initialisation de CSRMatrix échouée\n");
        free(nnz_per_row);
        return NULL;
    }

    csr->row_ptr[0] = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        csr->row_ptr[i + 1] = csr->row_ptr[i] + nnz_per_row[i];
    }

    /* Second parcours : remplissage CSR. */
    int* current_pos = (int*)calloc(num_rows, sizeof(int));
    if (!current_pos) {
        fprintf(stderr, "Erreur : Allocation de current_pos échouée\n");
        free_CSRMatrix(csr);
        free(csr);
        free(nnz_per_row);
        return NULL;
    }

    for (size_t i = 0; i < num_rows; ++i) {
        current_pos[i] = csr->row_ptr[i];
    }

    for (size_t i = 0; i < num_rows; i += BLOCK_SIZE) {
        size_t i_end = i + BLOCK_SIZE < num_rows ? i + BLOCK_SIZE : num_rows;
        for (size_t j = 0; j < num_cols; j += BLOCK_SIZE) {
            size_t j_end = j + BLOCK_SIZE < num_cols ? j + BLOCK_SIZE : num_cols;
            for (size_t ii = i; ii < i_end; ++ii) {
                for (size_t jj = j; jj < j_end; ++jj) {
                    if (fabs(dense->data[ii][jj]) > EPSILON) {
                        int k = current_pos[ii]++;
                        csr->values[k] = dense->data[ii][jj];
                        csr->col_index[k] = jj;
                    }
                }
            }
        }
    }

    free(current_pos);
    free(nnz_per_row);
    return csr;
}

CSR3DTensor* tensor_to_csr(Tensor3D* tensor) {
    if (tensor == NULL) {
        fprintf(stderr, "Erreur : tenseur d'entree NULL\n");
        return NULL;
    }
    CSR3DTensor* csr_tensor = init_CSR3DTensor(tensor->num_slices, tensor->rows, tensor->cols, 0);
    if (!csr_tensor) {
        return NULL;
    }

    for (int i = 0; i < tensor->num_slices; i++) {
        CSRMatrix* csr_slice = dense_to_csr(&tensor->slices[i]);
        if (csr_slice == NULL) {
            fprintf(stderr, "Erreur : conversion CSR echouee pour la tranche %d\n", i);
            free_CSR3DTensor(csr_tensor);
            return NULL;
        }

        free(csr_tensor->slices[i].values);
        free(csr_tensor->slices[i].col_index);
        free(csr_tensor->slices[i].row_ptr);

        csr_tensor->slices[i] = *csr_slice;
        free(csr_slice);
    }

    return csr_tensor;
}

Matrix* get_matrix(int n, int m, unsigned int seed) {
    if (n <= 0 || m <= 0) {
        fprintf(stderr, "Erreur : Dimensions invalides (%d x %d)\n", n, m);
        return NULL;
    }
    srand(seed);

    Matrix* matrix = init_Matrix(n, m);
    if (!matrix) {
        fprintf(stderr, "Erreur : Échec allocation structure Matrix\n");
        return NULL;
    }
    if (!matrix->data) {
        fprintf(stderr, "Erreur : Échec allocation données Matrix\n");
        free(matrix);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (!matrix->data[i]) {
            fprintf(stderr, "Erreur : Ligne %d non allouée\n", i);
            free_Matrix(matrix);
            return NULL;
        }
    }

    const double rand_max_inv = 1.0 / (RAND_MAX + 1e-15);

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int i_end = (i + BLOCK_SIZE) < n ? (i + BLOCK_SIZE) : n;
        
        for (int j = 0; j < m; j += BLOCK_SIZE) {
            int j_end = (j + BLOCK_SIZE) < m ? (j + BLOCK_SIZE) : m;

            for (int ii = i; ii < i_end; ++ii) {
                double* row = matrix->data[ii];
                for (int jj = j; jj < j_end; ++jj) {
                    row[jj] = ((double)rand()) * rand_max_inv;
                }
            }
        }
    }

    return matrix;
}

void log_info(int itr, double fit, double fitchange, double secs) {
    printf("INFO : RESCAL : [%3d] fit: %0.5f | delta: %7.1e | secs: %.5f\n", itr, fit, fitchange, secs);
}

void matrix_vector_multiply(Matrix* A, double* v, double* result) {
    if (!A || !A->data || A->rows <= 0 || A->cols <= 0 || !v || !result) {
        fprintf(stderr, "Erreur : Arguments invalides\n");
        return;
    }

    const int rows = A->rows;
    const int cols = A->cols;
    
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
    }

    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        const int i_end = (i + BLOCK_SIZE) < rows ? (i + BLOCK_SIZE) : rows;
        
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            const int j_end = (j + BLOCK_SIZE) < cols ? (j + BLOCK_SIZE) : cols;
            
            for (int ii = i; ii < i_end; ++ii) {
                const double* a_row = A->data[ii];
                double sum = 0.0;
                
                for (int jj = j; jj < j_end; ++jj) {
                    sum += a_row[jj] * v[jj];
                }
                result[ii] += sum;
            }
        }
    }
}

void normalize_vector(double* v, int n) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    if (norm < EPSILON) {
        return;
    }
    for (int i = 0; i < n; i++) {
        v[i] /= norm;
    }
}

double dot_product(double* v1, double* v2, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

Matrix* eigsh(Matrix* A, int k, double** eigenvalues) {
    if (!A || !A->data || A->rows <= 0 || A->rows != A->cols || k <= 0 || k > A->rows || !eigenvalues || !*eigenvalues) {
        fprintf(stderr, "Erreur : Arguments invalides pour eigsh\n");
        return NULL;
    }

    int n = A->rows;
    double* v = (double*)malloc(n * sizeof(double));
    double* Av = (double*)malloc(n * sizeof(double));
    if (!v || !Av) {
        fprintf(stderr, "Erreur : Allocation de v ou Av échouée\n");
        free(v);
        free(Av);
        return NULL;
    }

    Matrix* eigenvectors = init_Matrix(n, k);
    if (!eigenvectors || !eigenvectors->data) {
        fprintf(stderr, "Erreur : Initialisation de eigenvectors échouée\n");
        free(v);
        free(Av);
        free_Matrix(eigenvectors);
        return NULL;
    }

    for (int l = 0; l < k; l++) {
        for (int i = 0; i < n; i++) {
            v[i] = (double)rand() / RAND_MAX;
        }
        normalize_vector(v, n);
        if (fabs(dot_product(v, v, n)) < EPSILON) {
            fprintf(stderr, "Erreur : Vecteur initial nul pour l=%d\n", l);
            free(v);
            free(Av);
            free_Matrix(eigenvectors);
            return NULL;
        }

        for (int iter = 0; iter < MAX_ITER; iter++) {
            matrix_vector_multiply(A, v, Av);

            for (int j = 0; j < l; j++) {
                double proj = dot_product(Av, eigenvectors->data[j], n);
                for (int i = 0; i < n; i += BLOCK_SIZE) {
                    int i_end = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
                    for (int ii = i; ii < i_end; ++ii) {
                        Av[ii] -= proj * eigenvectors->data[ii][j];
                    }
                }
            }

            normalize_vector(Av, n);
            if (fabs(dot_product(Av, Av, n)) < EPSILON) {
                fprintf(stderr, "Erreur : Vecteur Av nul pour l=%d, iter=%d\n", l, iter);
                break;
            }

            double diff = 0.0;
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int i_end = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
                for (int ii = i; ii < i_end; ++ii) {
                    diff += fabs(Av[ii] - v[ii]);
                }
            }
            if (diff < EPSILON) {
                break;
            }

            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int i_end = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
                for (int ii = i; ii < i_end; ++ii) {
                    v[ii] = Av[ii];
                }
            }
        }

        matrix_vector_multiply(A, v, Av);
        (*eigenvalues)[l] = dot_product(v, Av, n);

        for (int i = 0; i < n; i += BLOCK_SIZE) {
            int i_end = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
            for (int ii = i; ii < i_end; ++ii) {
                eigenvectors->data[ii][l] = v[ii];
            }
        }
    }

    free(v);
    free(Av);
    return eigenvectors;
}

Matrix* add_matrices(Matrix* A, Matrix* B) {
    if (!A || !B || !A->data || !B->data || A->rows != B->rows || A->cols != B->cols ||
        A->rows <= 0 || A->cols <= 0) {
        fprintf(stderr, "Erreur : Matrices invalides ou incompatibles\n");
        return NULL;
    }

    Matrix* C = init_Matrix(A->rows, A->cols);
    if (!C || !C->data) {
        fprintf(stderr, "Erreur : Initialisation de la matrice résultat échouée\n");
        free_Matrix(C);
        return NULL;
    }

    for (int i = 0; i < A->rows; ++i) {
        if (!A->data[i] || !B->data[i] || !C->data[i]) {
            fprintf(stderr, "Erreur : Ligne %d invalide\n", i);
            free_Matrix(C);
            return NULL;
        }
    }

    for (int i = 0; i < A->rows; i += BLOCK_SIZE) {
        int i_end = i + BLOCK_SIZE < A->rows ? i + BLOCK_SIZE : A->rows;
        for (int j = 0; j < A->cols; j += BLOCK_SIZE) {
            int j_end = j + BLOCK_SIZE < A->cols ? j + BLOCK_SIZE : A->cols;
            for (int ii = i; ii < i_end; ++ii) {
                double* A_row = A->data[ii];
                double* B_row = B->data[ii];
                double* C_row = C->data[ii];
                for (int jj = j; jj < j_end; ++jj) {
                    C_row[jj] = A_row[jj] + B_row[jj];
                }
            }
        }
    }

    return C;
}

Matrix* T_matrix(Matrix* A) {
    if (!A || !A->data || A->rows <= 0 || A->cols <= 0) {
        fprintf(stderr, "Erreur : Matrice d'entrée invalide\n");
        return NULL;
    }

    Matrix* T = init_Matrix(A->cols, A->rows);
    if (!T || !T->data) {
        fprintf(stderr, "Erreur : Allocation échouée\n");
        free_Matrix(T);
        return NULL;
    }

    const int rows = A->rows;
    const int cols = A->cols;

    for (int i = 0; i < rows; i++) {
        if (!A->data[i]) {
            fprintf(stderr, "Erreur : Ligne %d de A invalide\n", i);
            free_Matrix(T);
            return NULL;
        }
    }
    for (int j = 0; j < cols; j++) {
        if (!T->data[j]) {
            fprintf(stderr, "Erreur : Ligne %d de T invalide\n", j);
            free_Matrix(T);
            return NULL;
        }
    }

    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        const int i_end = (i + BLOCK_SIZE) < rows ? (i + BLOCK_SIZE) : rows;
        
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            const int j_end = (j + BLOCK_SIZE) < cols ? (j + BLOCK_SIZE) : cols;
            
            for (int ii = i; ii < i_end; ++ii) {
                const double* a_row = A->data[ii];
                
                for (int jj = j; jj < j_end; ++jj) {
                    T->data[jj][ii] = a_row[jj];
                }
            }
        }
    }

    return T;
}

double LocalHypot(double a, double b) {
    double xabs = fabs(a);
    double yabs = fabs(b);
    double min, max;

    if (xabs < yabs) {
        min = xabs;
        max = yabs;
    } else {
        min = yabs;
        max = xabs;
    }

    if (min == 0) {
        return max;
    } else {
        double u = min / max;
        return max * sqrt(1 + u * u);
    }
}

void FreeSVDResult(SVDResult* svd) {
    if (!svd) return;

    if (svd->U.data && svd->U.data[0]) {
        free(svd->U.data[0]);
        free(svd->U.data);
        svd->U.data = NULL;
    }

    if (svd->S) {
        free(svd->S);
        svd->S = NULL;
    }

    if (svd->Vt.data && svd->Vt.data[0]) {
        free(svd->Vt.data[0]);
        free(svd->Vt.data);
        svd->Vt.data = NULL;
    }

    free(svd);
}

Tensor3D* init_tensor3D(const int num_slices, const int rows, const int cols) {
    if (num_slices <= 0 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "Erreur : Dimensions invalides\n");
        return NULL;
    }

    Tensor3D* tensor = (Tensor3D*)malloc(sizeof(Tensor3D));
    if (!tensor) {
        fprintf(stderr, "Erreur : Allocation du tensor échouée\n");
        return NULL;
    }

    tensor->num_slices = num_slices;
    tensor->rows = rows;
    tensor->cols = cols;

    tensor->slices = (Matrix*)malloc(num_slices * sizeof(Matrix));
    if (!tensor->slices) {
        fprintf(stderr, "Erreur : Allocation des slices échouée\n");
        free(tensor);
        return NULL;
    }

    double* global_data = (double*)calloc(num_slices * rows * cols, sizeof(double));
    if (!global_data) {
        fprintf(stderr, "Erreur : Allocation des données échouée\n");
        free(tensor->slices);
        free(tensor);
        return NULL;
    }

    for (int i = 0; i < num_slices; i++) {
        tensor->slices[i].rows = rows;
        tensor->slices[i].cols = cols;
        
        tensor->slices[i].data = (double**)malloc(rows * sizeof(double*));
        if (!tensor->slices[i].data) {
            fprintf(stderr, "Erreur : Allocation des lignes échouée\n");
            for (int k = 0; k < i; k++) {
                free(tensor->slices[k].data);
            }
            free(global_data);
            free(tensor->slices);
            free(tensor);
            return NULL;
        }

        double* slice_start = global_data + i * rows * cols;
        for (int j = 0; j < rows; j++) {
            tensor->slices[i].data[j] = slice_start + j * cols;
        }
    }

    return tensor;
}

void print_tensor3D(Tensor3D* tensor) {
    if (tensor == NULL) {
        printf("Le tenseur est NULL.\n");
        return;
    }

    printf("Tenseur3D (%d tranches, %d lignes, %d colonnes) :\n", tensor->num_slices, tensor->rows, tensor->cols);

    for (int i = 0; i < tensor->num_slices; i++) {
        printf("Tranche %d :\n", i);
        
        for (int j = 0; j < tensor->rows; j++) {
            for (int k = 0; k < tensor->cols; k++) {
                printf("%f ", tensor->slices[i].data[j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void free_tensor(Tensor3D* tensor) {
    if (!tensor) return;

    if (tensor->slices) {
        /* Les tranches partagent un bloc de données contigu. */
        double* global_data = NULL;
        if (tensor->num_slices > 0 && tensor->slices[0].data)
            global_data = tensor->slices[0].data[0];

        for (int i = 0; i < tensor->num_slices; i++) {
            if (tensor->slices[i].data) {
                free(tensor->slices[i].data);
                tensor->slices[i].data = NULL;
            }
        }

        if (global_data)
            free(global_data);

        free(tensor->slices);
    }

    free(tensor);
}

Matrix* init_Matrix(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour la structure de la matrice.\n");
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->data = (double**)malloc(rows * sizeof(double*));
    if (matrix->data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les lignes de la matrice.\n");
        free(matrix);
        return NULL;
    }

    matrix->data[0] = (double*)calloc(rows * cols, sizeof(double));
    if (matrix->data[0] == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les données de la matrice.\n");
        free(matrix->data);
        free(matrix);
        return NULL;
    }

    for (int i = 1; i < rows; ++i) {
        matrix->data[i] = matrix->data[0] + i * cols;
    }

    return matrix;
}

void free_Matrix(Matrix* matrix) {
    if (matrix != NULL) {
        if (matrix->data != NULL) {
            free(matrix->data[0]);
            free(matrix->data);
        }
        free(matrix);
    }
}

CSRMatrix* init_CSRMatrix(int rows, int cols, int nnz) {
    if (rows <= 0 || cols <= 0 || nnz < 0) {
        fprintf(stderr, "Erreur : dimensions invalides (rows=%d, cols=%d, nnz=%d).\n", rows, cols, nnz);
        return NULL;
    }

    CSRMatrix* matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!matrix) {
        fprintf(stderr, "Erreur d'allocation pour la structure CSRMatrix.\n");
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->nnz = nnz;

    matrix->values = (nnz > 0) ? (double*)calloc((size_t)nnz, sizeof(double)) : NULL;
    matrix->col_index = (nnz > 0) ? (int*)calloc((size_t)nnz, sizeof(int)) : NULL;
    matrix->row_ptr = (int*)calloc((size_t)rows + 1, sizeof(int));

    if ((nnz > 0 && (!matrix->values || !matrix->col_index)) || !matrix->row_ptr) {
        fprintf(stderr, "Erreur d'allocation pour les données CSRMatrix.\n");
        free_CSRMatrix(matrix);
        free(matrix);
        return NULL;
    }

    return matrix;
}

void init_CSRMatrix_in_place(CSRMatrix* matrix, int rows, int cols, int nnz) {
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->nnz = nnz;

    matrix->values = (double*)calloc(nnz, sizeof(double));
    matrix->col_index = (int*)calloc(nnz, sizeof(int));
    matrix->row_ptr = (int*)calloc(rows + 1, sizeof(int));

    if ((nnz > 0 && (!matrix->values || !matrix->col_index)) || !matrix->row_ptr) {
        fprintf(stderr, "Erreur d'allocation dans init_CSRMatrix_in_place\n");
        free_CSRMatrix(matrix);
    }
}

CSR3DTensor* init_CSR3DTensor(int num_slices, int rows, int cols, int nnz_per_slice) {
    CSR3DTensor* tensor = malloc(sizeof(CSR3DTensor));
    if (!tensor) return NULL;

    tensor->num_slices = num_slices;
    tensor->rows = rows;
    tensor->cols = cols;

    tensor->slices = malloc(num_slices * sizeof(CSRMatrix));

    if (!tensor->slices) {
        free(tensor);
        return NULL;
    }

    for (int i = 0; i < num_slices; ++i) {
        memset(&tensor->slices[i], 0, sizeof(CSRMatrix));
        init_CSRMatrix_in_place(&tensor->slices[i], rows, cols, nnz_per_slice);
        if (!tensor->slices[i].row_ptr ||
            (nnz_per_slice > 0 && (!tensor->slices[i].values || !tensor->slices[i].col_index))) {
            for (int j = 0; j <= i; j++) {
                free_CSRMatrix(&tensor->slices[j]);
            }
            free(tensor->slices);
            free(tensor);
            return NULL;
        }
    }

    return tensor;
}

void array_init(Array *a, size_t initial_size) {
    if (initial_size == 0) initial_size = 1;  /* capacite minimale de 1 */

    a->data = (double *)malloc(initial_size * sizeof(double));
    if (a->data == NULL) {
        fprintf(stderr, "Erreur : echec de l allocation memoire pour le tableau\n");
        a->size = 0;
        a->capacity = 0;
        return;
    }

    a->size = 0;
    a->capacity = initial_size;
}

void array_append(Array *a, double value) {
    if (a->size == a->capacity) {
        size_t new_capacity = a->capacity > 0 ? a->capacity * 2 : 1;
        double* new_data = (double *)realloc(a->data, new_capacity * sizeof(double));

        if (new_data == NULL) {
            fprintf(stderr, "Erreur : échec de la réallocation de mémoire pour le tableau\n");
            free(a->data);
            exit(EXIT_FAILURE);
        }

        a->data = new_data;
        a->capacity = new_capacity;
    }

    a->data[a->size++] = value;
}

void array_free(Array *a) {
    if (a == NULL) {
        return;
    }
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}

resultat init_resultat(int rows, int cols, int rank, int num_slices, size_t array_size) {
    resultat res;

    res.A = init_Matrix(rows, rank);

    res.R = init_tensor3D(num_slices, rows, cols);

    res.f = 0.0;

    res.itr = 0;

    array_init(&(res.exectimes), array_size);

    return res;
}

void free_resultat(resultat *res) {
    if (!res ) return;
    free_Matrix(res->A);
    free_tensor(res->R);
    array_free(&(res->exectimes));
}

void free_CSRMatrix(CSRMatrix* matrix) {
    if (matrix == NULL) return;

    if (matrix->values != NULL) {
        free(matrix->values);
        matrix->values = NULL;
    }
    
    if (matrix->col_index != NULL) {
        free(matrix->col_index);
        matrix->col_index = NULL;
    }
    
    if (matrix->row_ptr != NULL) {
        free(matrix->row_ptr);
        matrix->row_ptr = NULL;
    }

}

void free_CSR3DTensor(CSR3DTensor* tensor) {
    if (!tensor) return;

    if (tensor->slices) {
        for (int i = 0; i < tensor->num_slices; ++i) {
            free_CSRMatrix(&tensor->slices[i]);
        }
        free(tensor->slices);
    }
    free(tensor);
}


Matrix* copy_Matrix(Matrix* src) {
    if (!src) {
        fprintf(stderr, "Erreur : matrice source NULL\n");
        return NULL;
    }

    Matrix* dest = init_Matrix(src->rows, src->cols);
    if (!dest) return NULL;

    for (int i = 0; i < src->rows; i++) {
        memcpy(dest->data[i], src->data[i], src->cols * sizeof(double));
    }
    return dest;
}

Tensor3D* copy_Tensor3D(Tensor3D* src) {
    if (src == NULL) {
        fprintf(stderr, "Erreur : tenseur source NULL\n");
        return NULL;
    }

    Tensor3D* dest = init_tensor3D(src->num_slices, src->rows, src->cols);
    if (!dest) return NULL;

    for (int i = 0; i < src->num_slices; ++i) {
        for (int j = 0; j < src->rows; ++j) {
            memcpy(dest->slices[i].data[j],
                   src->slices[i].data[j],
                   src->cols * sizeof(double));
        }
    }

    return dest;
}

Matrix* dot(Matrix* A, Matrix* B) {
    if (!A || !B) {
        fprintf(stderr, "Erreur : matrices NULL dans dot\n");
        return NULL;
    }
    if (A->cols != B->rows) {
        fprintf(stderr, "Erreur : dimensions incompatibles dans dot\n");
        return NULL;
    }

    int M = A->rows;
    int K = A->cols;
    int N = B->cols;

    Matrix* C = init_Matrix(M, N);
    if (!C || !C->data) {
        fprintf(stderr, "Erreur : allocation échouée dans dot\n");
        free_Matrix(C);
        return NULL;
    }

    int p = g_num_threads;
    if (p > M) {
        p = M;
    }
    if (p < 1) {
        p = 1;
    }

    if (p == 1) {
        DotThreadArgs args = {
            A->data, B->data, C->data, M, K, N, 0, M
        };
        thread_block_dot(&args);
        return C;
    }

    pthread_t* threads = malloc((size_t)p * sizeof(pthread_t));
    DotThreadArgs* args = malloc((size_t)p * sizeof(DotThreadArgs));
    if (!threads || !args) {
        free(threads);
        free(args);
        free_Matrix(C);
        return NULL;
    }

    int base_rows = M / p;
    int remainder = M % p;
    int current_row = 0;

    for (int t = 0; t < p; t++) {
        int rows_for_t = base_rows + (t < remainder ? 1 : 0);
        args[t].A_data    = A->data;
        args[t].B_data    = B->data;
        args[t].C_data    = C->data;
        args[t].M         = M;
        args[t].K         = K;
        args[t].N         = N;
        args[t].row_start = current_row;
        args[t].row_end   = current_row + rows_for_t;
        current_row      += rows_for_t;

        pthread_create(&threads[t], NULL, thread_block_dot, &args[t]);
    }

    for (int t = 0; t < p; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);

    return C;
}

int is_nonzero_csr(CSRMatrix* X, int row, int col) {
    if (!X || row < 0 || row >= X->rows || col < 0 || col >= X->cols) {
        return 0;
    }
    for (int idx = X->row_ptr[row]; idx < X->row_ptr[row + 1]; idx++) {
        if (X->col_index[idx] == col) {
            return 1;
        }
    }
    return 0;
}

Matrix* eye(int n) {
    Matrix* matrix = init_Matrix(n, n);
    for (int i = 0; i < n; ++i) {
        matrix->data[i][i] = 1.0;
    }

    return matrix;
}

Matrix* multiply_by_scalar(Matrix* matrix, double scalar) {
    if (matrix == NULL) {
        return NULL;
    }

    Matrix* result = init_Matrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            result->data[i][j] = matrix->data[i][j] * scalar;
        }
    }

    return result;
}

/* Résolution AX = B par Gauss-Jordan avec pivot partiel. */
Matrix* solve(Matrix* A, Matrix* B) {
    if (!A || !B || A->rows != A->cols || A->cols != B->rows) {
        fprintf(stderr, "Erreur: Matrices invalides ou dimensions incompatibles\n");
        return NULL;
    }
    int n = A->rows;
    int k = B->cols;
    int width = n + k;

    /* Tampon temporaire pour l'échange de lignes. */
    double* row_buf = (double*)malloc(width * sizeof(double));
    if (!row_buf) return NULL;

    /* Tentatives avec régularisation Tikhonov croissante si pivot nul */
    double reg_vals[] = {0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2};
    int n_reg = (int)(sizeof(reg_vals) / sizeof(reg_vals[0]));

    for (int attempt = 0; attempt < n_reg; attempt++) {
        double reg = reg_vals[attempt];

        Matrix* aug = init_Matrix(n, width);
        if (!aug) { free(row_buf); return NULL; }

        /* Remplissage [A + reg*I | B] */
        for (int i = 0; i < n; i++) {
            double* ar = aug->data[i];
            double* Ar = A->data[i];
            double* Br = B->data[i];
            for (int j = 0; j < n; j++) ar[j] = Ar[j];
            ar[i] += reg;
            for (int j = 0; j < k; j++) ar[n + j] = Br[j];
        }

        int singular = 0;
        for (int i = 0; i < n; i++) {
            /* Pivot partiel. */
            int maxRow = i;
            double maxVal = fabs(aug->data[i][i]);
            for (int r = i + 1; r < n; r++) {
                double v = fabs(aug->data[r][i]);
                if (v > maxVal) { maxVal = v; maxRow = r; }
            }

            if (maxRow != i) {
                memcpy(row_buf,           aug->data[i],      width * sizeof(double));
                memcpy(aug->data[i],      aug->data[maxRow], width * sizeof(double));
                memcpy(aug->data[maxRow], row_buf,           width * sizeof(double));
            }

            double pivot = aug->data[i][i];
            if (fabs(pivot) < 1e-14) { singular = 1; break; }

            double inv_pivot = 1.0 / pivot;
            double* pr = aug->data[i];
            for (int j = 0; j < width; j++) pr[j] *= inv_pivot;

            for (int r = 0; r < n; r++) {
                if (r == i) continue;
                double* row = aug->data[r];
                double factor = row[i];
                if (fabs(factor) < 1e-15) continue;
                for (int j = 0; j < width; j++)
                    row[j] -= factor * pr[j];
            }
        }

        if (singular) {
            free_Matrix(aug);
            if (attempt < n_reg - 1)
                fprintf(stderr, "Warn : solve() pivot nul (reg=%.0e), retente avec reg=%.0e\n",
                        reg, reg_vals[attempt + 1]);
            continue;
        }

        Matrix* X = init_Matrix(n, k);
        if (!X) { free_Matrix(aug); free(row_buf); return NULL; }
        for (int i = 0; i < n; i++) {
            double* ar = aug->data[i];
            double* xr = X->data[i];
            for (int j = 0; j < k; j++) xr[j] = ar[n + j];
        }
        free_Matrix(aug);
        free(row_buf);
        return X;
    }

    fprintf(stderr, "Erreur : solve() matrice irreductiblement singuliere.\n");
    free(row_buf);
    return NULL;
}
Matrix* MatrixAugmentWithIdentity(Matrix* mat) {
    int n = mat->rows;
    Matrix* augmented = init_Matrix(n, 2 * n);
    if (!augmented) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            augmented->data[i][j] = mat->data[i][j];
        augmented->data[i][i + n] = 1.0;
    }
    return augmented;
}

Matrix* Inverse(Matrix* mat) {
    if (!mat || mat->rows != mat->cols) {
        fprintf(stderr, "Erreur: Inverse matrice non carree ou NULL\n");
        return NULL;
    }
    int n = mat->rows;
    Matrix* aug = MatrixAugmentWithIdentity(mat);
    if (!aug) return NULL;

    double* row_buf = (double*)malloc(2 * n * sizeof(double));
    if (!row_buf) { free_Matrix(aug); return NULL; }

    for (int i = 0; i < n; i++) {
        /* Pivot partiel. */
        int    maxRow = i;
        double maxVal = fabs(aug->data[i][i]);
        for (int r = i + 1; r < n; r++) {
            double v = fabs(aug->data[r][i]);
            if (v > maxVal) { maxVal = v; maxRow = r; }
        }

        if (maxRow != i) {
            memcpy(row_buf,           aug->data[i],      2*n*sizeof(double));
            memcpy(aug->data[i],      aug->data[maxRow], 2*n*sizeof(double));
            memcpy(aug->data[maxRow], row_buf,           2*n*sizeof(double));
        }

        double diag = aug->data[i][i];
        if (fabs(diag) < 1e-10) {
            fprintf(stderr, "Erreur: Inverse pivot nul — matrice singuliere\n");
            free_Matrix(aug); free(row_buf);
            return NULL;
        }

        double inv_diag = 1.0 / diag;
        for (int j = 0; j < 2*n; j++)
            aug->data[i][j] *= inv_diag;

        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = aug->data[k][i];
                if (fabs(factor) < 1e-15) continue;
            for (int j = 0; j < 2*n; j++)
                aug->data[k][j] -= factor * aug->data[i][j];
        }
    }

    Matrix* inv = init_Matrix(n, n);
    if (!inv) { free_Matrix(aug); free(row_buf); return NULL; }
    for (int i = 0; i < n; i++)
        memcpy(inv->data[i], aug->data[i] + n, n * sizeof(double));

    free_Matrix(aug);
    free(row_buf);
    return inv;
}

double* kron(double *vec1, int len1, double *vec2, int len2, int *resultLen) {
    *resultLen = len1 * len2;

    double *result = (double *)malloc(*resultLen * sizeof(double));
    if (result == NULL) {
        *resultLen = 0;
        return NULL;
    }

    for (int i = 0; i < len1; i++) {
        for (int j = 0; j < len2; j++) {
            result[i * len2 + j] = vec1[i] * vec2[j];
        }
    }
    
    return result;
}

Matrix* reshape(double* vector, int n) {
    if (vector == NULL) {
        fprintf(stderr, "Erreur : vecteur d'entrée NULL\n");
        return NULL;
    }

    Matrix* mat = init_Matrix(n, n);
    if (mat == NULL) {
        fprintf(stderr, "Erreur : échec de l'allocation de la matrice\n");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat->data[i][j] = vector[i * n + j];
        }
    }

    return mat;
}

Matrix* soustraction(Matrix* A, Matrix* B) {
    if (!A || !B) {
        fprintf(stderr, "Erreur : l'une des matrices est NULL.\n");
        return NULL;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Erreur : dimensions incompatibles.\n");
        return NULL;
    }

    const int n = A->rows;
    const int m = A->cols;
    
    Matrix* result = init_Matrix(n, m);
    if (!result || !result->data) {
        fprintf(stderr, "Erreur : allocation échouée.\n");
        free_Matrix(result);
        return NULL;
    }

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        const int i_end = (i + BLOCK_SIZE) < n ? (i + BLOCK_SIZE) : n;
        
        for (int j = 0; j < m; j += BLOCK_SIZE) {
            const int j_end = (j + BLOCK_SIZE) < m ? (j + BLOCK_SIZE) : m;
            
            for (int ii = i; ii < i_end; ii++) {
                double* a_row = A->data[ii];
                double* b_row = B->data[ii];
                double* r_row = result->data[ii];
                
                for (int jj = j; jj < j_end; jj++) {
                    r_row[jj] = a_row[jj] - b_row[jj];
                }
            }
        }
    }

    return result;
}

Matrix* dot_csr(CSRMatrix* A, Matrix* B) {
    if (!A || !B || A->cols != B->rows) {
        fprintf(stderr, "Erreur : dimensions incompatibles ou matrices NULL\n");
        return NULL;
    }

    const int M = A->rows;
    const int N = B->cols;
    Matrix* C = init_Matrix(M, N);
    if (!C || !C->data) {
        fprintf(stderr, "Erreur : allocation échouée\n");
        free_Matrix(C);
        return NULL;
    }

    int p = g_num_threads;
    if (p > M) {
        p = M;
    }
    if (p < 1) {
        p = 1;
    }

    if (p == 1) {
        DotCsrThreadArgs args = { A, B, C, 0, M };
        thread_dot_csr(&args);
        return C;
    }

    pthread_t* threads = malloc((size_t)p * sizeof(pthread_t));
    DotCsrThreadArgs* args = malloc((size_t)p * sizeof(DotCsrThreadArgs));
    if (!threads || !args) {
        free(threads);
        free(args);
        free_Matrix(C);
        return NULL;
    }

    for (int t = 0; t < p; t++) {
        int row_start = 0;
        int row_end = 0;
        distribute_rows(M, p, t, &row_start, &row_end);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        pthread_create(&threads[t], NULL, thread_dot_csr, &args[t]);
    }

    for (int t = 0; t < p; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
    return C;
}

CSRMatrix* T_csr(CSRMatrix* A) {
    if (!A) {
        return NULL;
    }
    
    CSRMatrix* AT = init_CSRMatrix(A->cols, A->rows, A->nnz);
    if (!AT) return NULL;

    for (int i = 0; i <= A->cols; i++) {
        AT->row_ptr[i] = 0;
    }

    for (int i = 0; i < A->nnz; i++) {
        AT->row_ptr[A->col_index[i] + 1]++;
    }

    for (int i = 0; i < A->cols; i++) {
        AT->row_ptr[i + 1] += AT->row_ptr[i];
    }

    int* temp = (int*)malloc(A->cols * sizeof(int));
    if (!temp) {
        free_CSRMatrix(AT);
        free(AT);
        return NULL;
    }

    for (int i = 0; i < A->cols; i++) {
        temp[i] = AT->row_ptr[i];
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            int col = A->col_index[j];
            int dest_pos = temp[col];

            AT->values[dest_pos] = A->values[j];
            AT->col_index[dest_pos] = i;

            temp[col]++;
        }
    }

    free(temp);
    return AT;
}

Matrix* dot_csr1(Matrix* A, CSRMatrix* B) {
    if (!A || !B || A->cols != B->rows) {
        fprintf(stderr, "Erreur : dimensions incompatibles\n");
        return NULL;
    }

    const int M = A->rows;
    const int N = B->cols;
    Matrix* C = init_Matrix(M, N);
    if (!C) return NULL;

    int p = g_num_threads;
    if (p > M) {
        p = M;
    }
    if (p < 1) {
        p = 1;
    }

    if (p == 1) {
        DotCsr1ThreadArgs args = { A, B, C, 0, M };
        thread_dot_csr1(&args);
        return C;
    }

    pthread_t* threads = malloc((size_t)p * sizeof(pthread_t));
    DotCsr1ThreadArgs* args = malloc((size_t)p * sizeof(DotCsr1ThreadArgs));
    if (!threads || !args) {
        free(threads);
        free(args);
        free_Matrix(C);
        return NULL;
    }

    for (int t = 0; t < p; t++) {
        int row_start = 0;
        int row_end = 0;
        distribute_rows(M, p, t, &row_start, &row_end);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        pthread_create(&threads[t], NULL, thread_dot_csr1, &args[t]);
    }

    for (int t = 0; t < p; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
    return C;
}

void printMatrix(Matrix* matrix) {
    if (matrix == NULL) {
        printf("Erreur : matrice NULL\n");
        return;
    }

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

void save_matrix_to_file(Matrix* matrix, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Erreur lors de l'ouverture du fichier pour l'enregistrement.\n");
        return;
    }

    fprintf(file, "%d %d\n", matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            fprintf(file, "%.10f ", matrix->data[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrice enregistrée avec succès dans %s\n", filename);
}
void print_vector(double* vec, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

int count_zeros(double* vec, int size, double tolerance) {
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (fabs(vec[i]) < tolerance) {
            count++;
        }
    }
    return count;
}

Matrix* load_matrix_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur : ouverture du fichier");
        return NULL;
    }

    int rows, cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        perror("Erreur : lecture des dimensions");
        fclose(file);
        return NULL;
    }

    Matrix* matrix = init_Matrix(rows, cols);
    if (matrix == NULL) {
        perror("Erreur : allocation de la matrice");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix->data[i][j]) != 1) {
                perror("Erreur : lecture des coefficients");
                free_Matrix(matrix);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return matrix;
}

int count_zero_elements(Matrix* matrix) {
    int count = 0;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            if (matrix->data[i][j] == 0.0) {
                count++;
            }
        }
    }
    return count;
}

void print_CSRMatrix(CSRMatrix* matrix) {
    if (matrix == NULL) {
        printf("La matrice CSR est NULL.\n");
        return;
    }
    
    printf("Matrice CSR (%d x %d) avec %d éléments non nuls :\n", matrix->rows, matrix->cols, matrix->nnz);
    
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; ++j) {
            if (fabs(matrix->values[j]) > 0.0) {
                printf("(%d, %d) ", i, matrix->col_index[j]);
            }
        }
    }
    printf("\n");
}

Matrix* prod(Matrix* A, Matrix* B) {
    if (A == NULL || B == NULL || A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Erreur : matrices invalides ou dimensions incompatibles\n");
        return NULL;
    }
    Matrix* result = init_Matrix(A->rows, A->cols);
    if (result == NULL) {
        fprintf(stderr, "Erreur : échec d'allocation mémoire\n");
        return NULL;
    }

    for (int i = 0; i < A->rows; i += BLOCK_SIZE) {
        for (int j = 0; j < A->cols; j += BLOCK_SIZE) {
            int i_end = (i + BLOCK_SIZE < A->rows) ? i + BLOCK_SIZE : A->rows;
            int j_end = (j + BLOCK_SIZE < A->cols) ? j + BLOCK_SIZE : A->cols;
            for (int ii = i; ii < i_end; ii++) {
                for (int jj = j; jj < j_end; jj++) {
                    result->data[ii][jj] = A->data[ii][jj] * B->data[ii][jj];
                }
            }
        }
    }
    return result;
}

void copy_matrix_data(Matrix* dest, Matrix* src) {
    if (dest == NULL || src == NULL) {
        fprintf(stderr, "Erreur : matrice source ou destination NULL\n");
        return;
    }
    if (dest->rows != src->rows || dest->cols != src->cols) {
        fprintf(stderr, "Erreur : dimensions incompatibles (%dx%d vs %dx%d)\n",
                dest->rows, dest->cols, src->rows, src->cols);
        return;
    }

    for (int i = 0; i < src->rows; ++i) {
        memcpy(dest->data[i], src->data[i], src->cols * sizeof(double));
    }
}

Matrix* create_scaled_identity(int size, double scalar) {
    Matrix* result = init_Matrix(size, size);
    for (int i = 0; i < size; i++) {
        result->data[i][i] = scalar;
    }
    return result;
}

int count_nonzero_elements(Tensor3D* tensor) {
    int count = 0;
    for (int k = 0; k < tensor->num_slices; k++) {
        Matrix* slice = &tensor->slices[k];
        for (int i = 0; i < slice->rows; i++) {
            for (int j = 0; j < slice->cols; j++) {
                if (slice->data[i][j] != 0) {
                    count++;
                }
            }
        }
    }
    return count;
}

void print_nonzero_count(Tensor3D* tensor) {
    int nonzero_count = count_nonzero_elements(tensor);
    printf("Nombre d'elements non nuls : %d\n", nonzero_count);
}
