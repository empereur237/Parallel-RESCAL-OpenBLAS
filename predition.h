/*
 * Auteur      : Projet RESCAL-ALS
 * Date        : 2026
 * Description : chargement des tenseurs, prédiction et métriques.
 */

#ifndef PREDITION_H
#define PREDITION_H

#include "utiles.h"
#include "rescal.h"


FileList* init_FileList(int initial_capacity);

void add_filename(FileList* list, const char* filename);

void free_FileList(FileList* list);

int compare_filenames(const void* a, const void* b);

Tensor3D* create_random_binary_tensor(int slices, int rows, int cols);

Tensor3D* load_tensor_from_directory(const char* directory_path);

Matrix* read_matrix_from_file(const char* filename);

Tensor3D* predict_rescal_als(Matrix* A, Tensor3D* R);

void normalize_predictions(Tensor3D* H, int e, int k);

void unravel_index(int idx, int rows, int slices, int* a, int* b, int* c);

Tensor3D* innerfold(Tensor3D* T, Tensor3D* P, int* mask_idx, int mask_length,
                    int rank, int maxIter, double conv,
                    double lambda_A, double lambda_R, double lambda_Z);

double compute_auc_pr(double* y_true, double* y_pred, int length);

double mean(const double* array, int size) ;

double stddev(const double* array, int size);

int compare_pairs(const void *a, const void *b);

void shuffle(int* array, int size);

double calculate_auc_pr(Tensor3D* T, int* target_idx, int target_length, Tensor3D* predicted_tensor);

#endif
