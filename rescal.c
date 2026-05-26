#include "rescal.h"

#define FIT_SAMPLE_BUDGET 10000

double _compute_fit(CSR3DTensor* X, Tensor3D* X1, Matrix* A, Tensor3D* R,
                   double lambda_A, double lambda_R) {
    if (!X || !X1 || !A || !R) {
        fprintf(stderr, "Erreur : Parametres invalides dans _compute_fit\n");
        return 0.0;
    }

    /* Norme de Frobenius de X via les valeurs CSR (evite de parcourir les zeros) */
    double sumNorm = 0.0;
    int observed_count = 0;
    for (int k = 0; k < X->num_slices; ++k)
        for (int i = 0; i < X->slices[k].nnz; ++i) {
            double val = X->slices[k].values[i];
            sumNorm += val * val;
            if (fabs(val) > EPSILON) {
                observed_count++;
            }
        }

    if (sumNorm < 1e-10) {
        fprintf(stderr, "Erreur : sumNorm trop faible (%f)\n", sumNorm);
        return 0.0;
    }

    double compute = 0.0;

    /*
     * Critere creux : on evalue l'erreur sur les entrees observees du tenseur
     * CSR. Cette forme evite explicitement la reconstruction dense A R_k A^T,
     * qui est impossible pour FB15k-237.
     */
    int sample_stride = 1;
    if (observed_count > FIT_SAMPLE_BUDGET) {
        sample_stride = observed_count / FIT_SAMPLE_BUDGET;
        if (sample_stride < 1) {
            sample_stride = 1;
        }
    }

    int seen = 0;
    int evaluated = 0;
    for (int k = 0; k < X->num_slices; ++k) {
        CSRMatrix* slice = &X->slices[k];
        Matrix* Rk = &R->slices[k];

        for (int row = 0; row < slice->rows; row++) {
            for (int p = slice->row_ptr[row]; p < slice->row_ptr[row + 1]; p++) {
                double observed = slice->values[p];
                if (fabs(observed) <= EPSILON) {
                    continue;
                }
                if ((seen++ % sample_stride) != 0) {
                    continue;
                }

                int col = slice->col_index[p];
                double predicted = 0.0;

                for (int i = 0; i < A->cols; i++) {
                    double left = A->data[row][i];
                    if (fabs(left) <= EPSILON) {
                        continue;
                    }

                    for (int j = 0; j < A->cols; j++) {
                        predicted += left * Rk->data[i][j] * A->data[col][j];
                    }
                }

                double diff = observed - predicted;
                compute += diff * diff;
                evaluated++;
            }
        }
    }

    if (evaluated > 0 && evaluated < observed_count) {
        compute *= (double)observed_count / (double)evaluated;
    }

    double norm_A = 0.0;
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            norm_A += A->data[i][j] * A->data[i][j];

    double norm_R = 0.0;
    for (int i = 0; i < R->num_slices; i++) {
        int sz = R->slices[i].rows * R->slices[i].cols;
        double* p = R->slices[i].data[0];
        for (int j = 0; j < sz; j++)
            norm_R += p[j] * p[j];
    }

    compute += lambda_A * norm_A + lambda_R * norm_R;
    double fit = 1.0 - (compute / sumNorm);

    return fit;
}

Tensor3D* _updateR(CSR3DTensor* X, Matrix* A, double lmbdaR, int rank) {
    Tensor3D* R = init_tensor3D(X->num_slices, rank, rank);
    if (!R) return NULL;

    /* Z_inv = (A^T A + lambda_R I)^-1. */
    Matrix* At  = T_matrix(A);
    Matrix* AtA = dot(At, A);
    for (int i = 0; i < rank; i++) AtA->data[i][i] += lmbdaR;

    Matrix* Z_inv = Inverse(AtA);
    if (!Z_inv) {
        fprintf(stderr, "Erreur: _updateR Inverse echouee\n");
        free_Matrix(At); free_Matrix(AtA); free_tensor(R);
        return NULL;
    }
    free_Matrix(AtA);

    /* R_k = Z_inv (A^T X_k A) Z_inv. */
    for (int k = 0; k < X->num_slices; k++) {
        Matrix* AtXk  = dot_csr1(At, &X->slices[k]);
        Matrix* AtXkA = dot(AtXk, A);
        free_Matrix(AtXk);

        Matrix* left = dot(Z_inv, AtXkA);
        free_Matrix(AtXkA);

        Matrix* Rk = dot(left, Z_inv);
        free_Matrix(left);

        copy_matrix_data(&R->slices[k], Rk);
        free_Matrix(Rk);
    }

    free_Matrix(At);
    free_Matrix(Z_inv);
    return R;
}

Tensor3D* _updateZ(CSR3DTensor* P, Matrix* A, double lambda_Z) {
    Tensor3D* Z = init_tensor3D(P->num_slices, P->cols, P->cols);
    if (P->num_slices == 0) {
        return Z;
    }

    Matrix* At = T_matrix(A);
    Matrix* AtA = dot(At, A);
    Matrix* reg = create_scaled_identity(A->cols, lambda_Z);
    Matrix* AtA_reg = add_matrices(AtA, reg);
    
    Matrix* pinvAt = Inverse(AtA_reg);
    Matrix* temp_pinv = dot(pinvAt, At);
    Matrix* pinvAt_transposed = T_matrix(temp_pinv);
    
    free_Matrix(At);
    free_Matrix(AtA);
    free_Matrix(reg);
    free_Matrix(AtA_reg);
    free_Matrix(pinvAt);
    free_Matrix(temp_pinv);
    
    int sparse_threshold = P->rows * P->cols * 3 / 100;

    for (int i = 0; i < P->num_slices; i++) {
        if (P->slices[i].nnz < sparse_threshold) {
            CSRMatrix* Pi_T = T_csr(&P->slices[i]);
            Matrix* Pi_T_csr = dot_csr(Pi_T, pinvAt_transposed);
            Matrix* transpose = T_matrix(Pi_T_csr);
            
            for (int j = 0; j < P->cols; j++) {
                memcpy(Z->slices[i].data[j], transpose->data[j], P->cols * sizeof(double));
            }
            
            free_Matrix(transpose);
            free_Matrix(Pi_T_csr);
            free_CSRMatrix(Pi_T);
            free(Pi_T);
        } else {
            Matrix* tempp = T_matrix(pinvAt_transposed);
            Matrix* temp = dot_csr1(tempp, &P->slices[i]);
            
            for (int j = 0; j < P->cols; j++) {
                memcpy(Z->slices[i].data[j], temp->data[j], P->cols * sizeof(double));
            }
            
            free_Matrix(tempp);
            free_Matrix(temp);
        }
    }
    
    free_Matrix(pinvAt_transposed);
    return Z;
}

Matrix* _updateA(CSR3DTensor* X, Matrix* A, Tensor3D* R, CSR3DTensor* P, Tensor3D* Z, double lambda_A, int rank) {
    Matrix* F = init_Matrix(X->rows, rank);
    Matrix* E = init_Matrix(rank, rank); 
    Matrix* At = T_matrix(A); 
    Matrix* AtA = dot(At, A);

    Matrix** Rt_slices = malloc(X->num_slices * sizeof(Matrix*));
    for (int i = 0; i < X->num_slices; i++) {
        Rt_slices[i] = T_matrix(&R->slices[i]);
    }

    for (int i = 0; i < X->num_slices; i++) {
        Matrix* ARt = dot(A, Rt_slices[i]);
        Matrix* XtARt = dot_csr(&X->slices[i], ARt);
        
        CSRMatrix* Xt = T_csr(&X->slices[i]);
        Matrix* XtA = dot_csr(Xt, A);
        Matrix* XtAR = dot(XtA, &R->slices[i]);

        Matrix* sum1 = add_matrices(XtARt, XtAR);
        Matrix* new_F = add_matrices(F, sum1);
        free_Matrix(F);
        F = new_F;

        Matrix* AtARt = dot(AtA, Rt_slices[i]);
        Matrix* R_AtARt = dot(&R->slices[i], AtARt);
        Matrix* Rt_AtA = dot(Rt_slices[i], AtA);
        Matrix* Rt_AtA_R = dot(Rt_AtA, &R->slices[i]);

        Matrix* sum2 = add_matrices(R_AtARt, Rt_AtA_R);
        Matrix* new_E = add_matrices(E, sum2);
        free_Matrix(E);
        E = new_E;

        free_Matrix(ARt); free_Matrix(XtARt);
        free_CSRMatrix(Xt); free(Xt); free_Matrix(XtA); free_Matrix(XtAR);
        free_Matrix(AtARt); free_Matrix(R_AtARt);
        free_Matrix(Rt_AtA); free_Matrix(Rt_AtA_R);
        free_Matrix(sum1); free_Matrix(sum2);
    }

    free_Matrix(AtA);
    free_Matrix(At);
    for (int i = 0; i < X->num_slices; i++) {
        free_Matrix(Rt_slices[i]);
    }
    free(Rt_slices);

    Matrix* I_ = create_scaled_identity(rank, lambda_A);

    if (P != NULL && P->num_slices > 0 && Z != NULL) {
        for (int i = 0; i < P->num_slices; i++) {
            Matrix* Zt_i = T_matrix(&Z->slices[i]);
            Matrix* PZt = dot_csr(&P->slices[i], Zt_i);
            
            Matrix* new_F = add_matrices(F, PZt);
            free_Matrix(F);
            F = new_F;
            
            Matrix* ZZt = dot(&Z->slices[i], Zt_i);
            Matrix* new_E = add_matrices(E, ZZt);
            free_Matrix(E);
            E = new_E;
            
            free_Matrix(Zt_i); free_Matrix(PZt); free_Matrix(ZZt);
        }
    }

    Matrix* Et = T_matrix(E);
    Matrix* E_final = add_matrices(I_, Et);
    Matrix* F_final = T_matrix(F);

    Matrix* E_f = Inverse(E_final);
    Matrix* A_new = dot(E_f, F_final);

    Matrix* result = T_matrix(A_new);
    
    free_Matrix(I_);  free_Matrix(Et); free_Matrix(E_f);
    free_Matrix(E_final);free_Matrix(E);
    free_Matrix(F); free_Matrix(F_final); free_Matrix(A_new);
    
    return result;
}

resultat rescal_als(Tensor3D* X1, int rank, char* init, int maxIter, double conv,
                   double lambda_A, double lambda_R, double lambda_Z, Tensor3D* P1, int dtype) {
    if (init == NULL) init = _DEF_INIT;
    if (maxIter <= 0) maxIter = _DEF_MAXITER;
    if (conv <= 0) conv = _DEF_CONV;
    if (lambda_A < 0) lambda_A = _DEF_LMBDA;
    if (lambda_R < 0) lambda_R = _DEF_LMBDA;
    if (lambda_Z < 0) lambda_Z = _DEF_LMBDA;
    if (dtype == 0) dtype = _DEF_TYPE;

    check_slices(X1);
    int n = X1->slices[0].rows;
    int k = X1->num_slices;

    CSR3DTensor* X = tensor_to_csr(X1);
    if (!X) {
        fprintf(stderr, "Erreur : conversion CSR echouee\n");
        return (resultat){0};
    }
    CSR3DTensor* P = (P1 != NULL) ? tensor_to_csr(P1) : NULL;

    static unsigned int seed_counter = 0;
    Matrix* A = NULL;

    if (strcmp(init, "random") == 0) {
        A = get_matrix(n, rank, rand() + seed_counter++);
    }
    else if (strcmp(init, "nvecs") == 0) {
        /* Initialisation spectrale par S = somme_k (X_k + X_k^T). */
        Matrix* S = init_Matrix(n, n);
        for (int i = 0; i < k; i++) {
            Matrix* Xt   = T_matrix(&X1->slices[i]);
            Matrix* sym  = add_matrices(&X1->slices[i], Xt);
            free_Matrix(Xt);
            Matrix* newS = add_matrices(S, sym);
            free_Matrix(S);
            free_Matrix(sym);
            S = newS;
        }
        SVDResult* svd = MatDecompSVD(S);
        free_Matrix(S);
        if (!svd) {
            fprintf(stderr, "Warn : SVD echouee pour nvecs, fallback random\n");
            A = get_matrix(n, rank, rand() + seed_counter++);
        } else {
            A = init_Matrix(n, rank);
            int use = (rank < svd->U.cols) ? rank : svd->U.cols;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < use; j++)
                    A->data[i][j] = svd->U.data[i][j];
            FreeSVDResult(svd);
        }
    }
    else {
        fprintf(stderr, "Erreur : initialisation de A non reconnue : %s\n", init);
        free_CSR3DTensor(X);
        if (P) free_CSR3DTensor(P);
        return (resultat){0};
    }

    if (!A) {
        fprintf(stderr, "Erreur : initialisation de A echouee\n");
        free_CSR3DTensor(X);
        if (P) free_CSR3DTensor(P);
        return (resultat){0};
    }

    Tensor3D* R = _updateR(X, A, lambda_R, rank);
    Tensor3D* Z = (P != NULL && P->num_slices > 0) ? _updateZ(P, A, lambda_Z) : NULL;

    double fit = 0.0, fitold = 0.0, fitchange = 0.0;
    Array exectimes;
    array_init(&exectimes, (size_t)maxIter);

    for (int itr = 0; itr < maxIter; ++itr) {
        clock_t iter_start = clock();
        fitold = fit;

        Matrix* new_A = _updateA(X, A, R, P, Z, lambda_A, rank);
        if (!new_A) { fprintf(stderr, "Warn : _updateA NULL a itr %d\n", itr); break; }
        free_Matrix(A);
        A = new_A;

        Tensor3D* new_R = _updateR(X, A, lambda_R, rank);
        if (!new_R) { fprintf(stderr, "Warn : _updateR NULL a itr %d\n", itr); break; }
        free_tensor(R);
        R = new_R;

        if (P != NULL && P->num_slices > 0) {
            Tensor3D* new_Z = _updateZ(P, A, lambda_Z);
            if (new_Z) { free_tensor(Z); Z = new_Z; }
        }

        fit = _compute_fit(X, X1, A, R, lambda_A, lambda_R);
        fitchange = fabs(fitold - fit);

        double iter_time = (double)(clock() - iter_start) / CLOCKS_PER_SEC;
        array_append(&exectimes, iter_time);
        log_info(itr, fit, fitchange, iter_time);

        if (itr > 0 && fitchange < conv) break;
    }

    resultat final = {0};
    final.A   = copy_Matrix(A);
    final.R   = copy_Tensor3D(R);
    final.f   = fit;
    final.itr = (int)exectimes.size;

    size_t n_t = exectimes.size > 0 ? exectimes.size : 1;
    array_init(&final.exectimes, n_t);
    for (size_t i = 0; i < exectimes.size; i++)
        array_append(&final.exectimes, exectimes.data[i]);

    free_Matrix(A);
    free_tensor(R);
    if (Z) free_tensor(Z);
    free_CSR3DTensor(X);
    if (P) free_CSR3DTensor(P);
    array_free(&exectimes);

    return final;
}
