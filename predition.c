/*
 * Auteur      : Projet RESCAL-ALS
 * Date        : 2026
 * Description : chargement des tenseurs, prédiction et métriques.
 */

#include "predition.h"
#include <assert.h>

#define _DEFAULT_SOURCE
//#define _POSIX_C_SOURCE 200809L

FileList* init_FileList(int initial_capacity) {
    if (initial_capacity <= 0) {
        initial_capacity = 10;
    }

    FileList* list = malloc(sizeof(FileList));
    if (!list) {
        fprintf(stderr, "Erreur : allocation de la liste de fichiers\n");
        return NULL;
    }

    list->filenames = malloc(initial_capacity * sizeof(char*));
    if (!list->filenames) {
        fprintf(stderr, "Erreur : allocation des noms de fichiers\n");
        free(list);
        return NULL;
    }

    list->count = 0;
    list->capacity = initial_capacity;
    return list;
}

void add_filename(FileList* list, const char* filename) {
    if (!list || !filename) {
        return;
    }

    if (list->count == list->capacity) {
        list->capacity *= 2;
        list->filenames = realloc(list->filenames, list->capacity * sizeof(char*));
        if (list->filenames == NULL) {
            fprintf(stderr, "Erreur : échec de l'allocation mémoire pour la liste de fichiers\n");
            exit(1);
        }
    }
    
    size_t len = strlen(filename) + 1;
    list->filenames[list->count] = malloc(len * sizeof(char));
    if (list->filenames[list->count] == NULL) {
        fprintf(stderr, "Erreur : échec de l'allocation mémoire pour le nom de fichier\n");
        exit(1);
    }
    
    strcpy(list->filenames[list->count], filename);
    list->count++;
}

void free_FileList(FileList* list) {
    if (!list) {
        return;
    }

    for (int i = 0; i < list->count; i++) {
        free(list->filenames[i]);
    }
    free(list->filenames);
    free(list);
}

int compare_filenames(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

Matrix* read_matrix_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Erreur : impossible d'ouvrir le fichier %s\n", filename);
        return NULL;
    }

    int rows, cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Erreur : impossible de lire les dimensions de la matrice dans le fichier %s\n", filename);
        fclose(file);
        return NULL;
    }

    Matrix* matrix = init_Matrix(rows, cols);
    if (matrix == NULL) {
        fprintf(stderr, "Erreur : échec de l'allocation mémoire pour la matrice\n");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix->data[i][j]) != 1) {
                fprintf(stderr, "Erreur : impossible de lire l'élément [%d, %d] dans le fichier %s\n", i, j, filename);
                fclose(file);
                free_Matrix(matrix);
                return NULL;
            }
        }
    }

    fclose(file);
    return matrix;
}

Tensor3D* load_tensor_from_directory(const char* directory_path) {
    DIR* dir = opendir(directory_path);
    if (dir == NULL) {
        fprintf(stderr, "Erreur : impossible d'ouvrir le répertoire %s\n", directory_path);
        return NULL;
    }

    FileList* file_list = init_FileList(10);
    if (!file_list) {
        closedir(dir);
        return NULL;
    }

    struct dirent* entry;
    char filepath[MAX_FILENAME];

    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".txt")) {
            snprintf(filepath, sizeof(filepath), "%s/%s", directory_path, entry->d_name);
            add_filename(file_list, filepath);
        }
    }
    closedir(dir);

    if (file_list->count == 0) {
        fprintf(stderr, "Erreur : aucun fichier .txt trouvé dans le répertoire %s\n", directory_path);
        free_FileList(file_list);
        return NULL;
    }

    /* Ordre déterministe des tranches. */
    qsort(file_list->filenames, file_list->count, sizeof(char*), compare_filenames);

    Matrix* first_matrix = read_matrix_from_file(file_list->filenames[0]);
    if (first_matrix == NULL) {
        free_FileList(file_list);
        return NULL;
    }

    Tensor3D* tensor = init_tensor3D(file_list->count, first_matrix->rows, first_matrix->cols);
    if (tensor == NULL) {
        fprintf(stderr, "Erreur : échec de l'allocation mémoire pour le tenseur\n");
        free_Matrix(first_matrix);
        free_FileList(file_list);
        return NULL;
    }

    for (int i = 0; i < first_matrix->rows; i++) {
        for (int j = 0; j < first_matrix->cols; j++) {
            tensor->slices[0].data[i][j] = first_matrix->data[i][j];
        }
    }
    free_Matrix(first_matrix);

    for (int k = 1; k < file_list->count; k++) {
        Matrix* temp_matrix = read_matrix_from_file(file_list->filenames[k]);
        if (temp_matrix == NULL) {
            fprintf(stderr, "Erreur : impossible de lire la matrice dans le fichier %s\n", file_list->filenames[k]);
            free_tensor(tensor);
            free_FileList(file_list);
            return NULL;
        }

        if (temp_matrix->rows != tensor->rows || temp_matrix->cols != tensor->cols) {
            fprintf(stderr, "Erreur : dimensions incompatibles dans %s\n", file_list->filenames[k]);
            free_Matrix(temp_matrix);
            free_tensor(tensor);
            free_FileList(file_list);
            return NULL;
        }

        for (int i = 0; i < temp_matrix->rows; i++) {
            for (int j = 0; j < temp_matrix->cols; j++) {
                tensor->slices[k].data[i][j] = temp_matrix->data[i][j];
            }
        }
        free_Matrix(temp_matrix);
    }

    free_FileList(file_list);
    return tensor;
}

Tensor3D* create_random_binary_tensor(int slices, int rows, int cols) {
    Tensor3D* tensor = init_tensor3D(slices, rows, cols);
    if (tensor == NULL) {
        return NULL;
    }

    srand(time(NULL));
    for (int i = 0; i < slices; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                tensor->slices[i].data[j][k] = (double)(rand() & 1);
            }
        }
    }

    return tensor;
}

Tensor3D* predict_rescal_als(Matrix* A, Tensor3D* R) {
    if (!A || !R) {
        return NULL;
    }

    int n = A->rows;
    Tensor3D* Pf = init_tensor3D(R->num_slices, n, n);
    if (!Pf) return NULL;
    for (int k = 0; k < R->num_slices; ++k) {
        Matrix* temp1 = T_matrix(A);
        Matrix* temp2 = dot(&R->slices[k], temp1);
        free_Matrix(temp1);
        Matrix* temp3 = dot(A, temp2);
        free_Matrix(temp2);
        copy_matrix_data(&Pf->slices[k], temp3);
        free_Matrix(temp3);
    }
    return Pf;
}

void normalize_predictions(Tensor3D* H, int e, int k) {
    if (!H) {
        return;
    }

    if (e > H->rows || e > H->cols || k > H->num_slices) {
        fprintf(stderr, "Erreur: dimensions invalides dans la normalisation du tenseur predit\n");
        return;
    }

    for (int a = 0; a < e; a++) {
        for (int b = 0; b < e; b++) {
            double nrm = 0.0;
            
            for (int c = 0; c < k; c++) {
                double val = H->slices[c].data[a][b];
                nrm += val * val;
            }
            nrm = sqrt(nrm);

            if (nrm != 0) {
                for (int c = 0; c < k; c++) {
                    H->slices[c].data[a][b] = round((H->slices[c].data[a][b] / nrm) * 1000.0) / 1000.0;
                }
            }
        }
    }
}

void unravel_index(int idx, int rows, int slices, int* a, int* b, int* c) {
    *c = idx % slices;
    int temp = idx / slices;
    *b = temp % rows;
    *a = temp / rows;
}

Tensor3D* innerfold(Tensor3D* T, Tensor3D* P, int* mask_idx, int mask_length,
                    int rank, int maxIter, double conv,
                    double lambda_A, double lambda_R, double lambda_Z) {
    Tensor3D* T_train = copy_Tensor3D(T);
    for (int i = 0; i < mask_length; i++) {
        int a, b, c;
        unravel_index(mask_idx[i], T->rows, T->num_slices, &a, &b, &c);
        T_train->slices[c].data[a][b] = 0.0;
    }
    resultat trained_model = rescal_als(T_train, rank, "random",
                                        maxIter, conv,
                                        lambda_A, lambda_R, lambda_Z,
                                        P, 0);
    if (!trained_model.A || !trained_model.R) {
        free_tensor(T_train);
        free_resultat(&trained_model);
        return NULL;
    }

    int n = trained_model.A->rows;
    Tensor3D* predicted_tensor = predict_rescal_als(trained_model.A, trained_model.R);
    if (predicted_tensor) {
        normalize_predictions(predicted_tensor, n, trained_model.R->num_slices);
    }
    free_tensor(T_train);
    free_resultat(&trained_model);
    return predicted_tensor;
}

int compare_pairs(const void *a, const void *b) {
    double diff = ((ScoreLabelPair*)b)->score - ((ScoreLabelPair*)a)->score;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

double compute_auc_pr(double* y_true, double* y_pred, int length) {
    if (!y_true || !y_pred || length <= 0) {
        return 0.0;
    }

    ScoreLabelPair* pairs = malloc(length * sizeof(ScoreLabelPair));
    if (!pairs) {
        return 0.0;
    }

    int total_positives = 0;
    for (int i = 0; i < length; i++) {
        pairs[i].score = y_pred[i];
        pairs[i].true_label = (int)(y_true[i] + 0.5);
        if (pairs[i].true_label == 1) total_positives++;
    }

    qsort(pairs, length, sizeof(ScoreLabelPair), compare_pairs);

    double* precision = malloc((length + 1) * sizeof(double));
    double* recall = malloc((length + 1) * sizeof(double));
    if (!precision || !recall) {
        free(pairs);
        free(precision);
        free(recall);
        return 0.0;
    }
    
    double tp = 0, fp = 0;
    int last_precision_idx = 0;
    
    for (int i = 0; i < length; i++) {
        if (pairs[i].true_label == 1) tp++;
        else fp++;
        
        if (i == length - 1 || pairs[i].score != pairs[i+1].score) {
            precision[last_precision_idx] = (tp + fp > 0) ? tp / (tp + fp) : 0;
            recall[last_precision_idx] = (total_positives > 0) ? tp / total_positives : 0;
            last_precision_idx++;
        }
    }
    
    /* Point initial de la courbe précision-rappel. */
    precision[last_precision_idx] = 1.0;
    recall[last_precision_idx] = 0.0;
    last_precision_idx++;

    for (int i = 0; i < last_precision_idx; i++) {
        for (int j = i + 1; j < last_precision_idx; j++) {
            if (recall[i] > recall[j]) {
                double temp = recall[i];
                recall[i] = recall[j];
                recall[j] = temp;
                temp = precision[i];
                precision[i] = precision[j];
                precision[j] = temp;
            }
        }
    }

    double auc_pr = 0.0;
    for (int i = 0; i < last_precision_idx - 1; i++) {
        auc_pr += (recall[i+1] - recall[i]) * (precision[i] + precision[i+1]) / 2.0;
    }

    free(pairs);
    free(precision);
    free(recall);

    return auc_pr;
}

double mean(const double* array, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum / size;
}

double stddev(const double* array, int size) {
    double m = mean(array, size);
    double sum_squared_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = array[i] - m;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / size);
}

void shuffle(int* array, int size) {
    if (size > 1) {
        for (int i = size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

double calculate_auc_pr(Tensor3D* T, int* target_idx, int target_length, Tensor3D* predicted_tensor) {
    if (!T || !target_idx || !predicted_tensor || target_length <= 0) {
        return 0.0;
    }

    double* y_true = malloc(target_length * sizeof(double));
    double* y_pred = malloc(target_length * sizeof(double));
    if (!y_true || !y_pred) {
        free(y_true);
        free(y_pred);
        return 0.0;
    }

    for (int i = 0; i < target_length; i++) {
        int a, b, c;
        unravel_index(target_idx[i], T->rows, T->num_slices, &a, &b, &c);
        y_true[i] = T->slices[c].data[a][b];
        y_pred[i] = predicted_tensor->slices[c].data[a][b];
    }

    double auc_pr = compute_auc_pr(y_true, y_pred, target_length);

    free(y_true);
    free(y_pred);
    return auc_pr;
}
