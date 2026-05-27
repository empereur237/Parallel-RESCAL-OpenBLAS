#ifndef UTILES_H
#define UTILES_H
#define _POSIX_C_SOURCE 200112L
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#ifndef RESCAL_BLAS_DEBUG
#define RESCAL_BLAS_DEBUG 0
#endif

#define MAX_ITER 100
#define EPSILON 1e-6
#define __version__ "0.4"
#define __als__ {'als'}
#define _DEF_MAXITER 100
#define _DEF_INIT "nvecs"
#define _DEF_CONV 1e-4
#define _DEF_LMBDA 10
#define _DEF_ATTR NULL
#define _DEF_NO_FIT 1e9
#define _DEF_FIT_METHOD 0
#define _DEF_ATTR_COUNT 0
#define _DEF_TYPE 1
#define MAX_FILENAME 1024
/* Taille de bloc pour le cache blocking -- calibree pour L1 cache */
/* Valeur recommandee : 32 a 64 selon la taille du cache L1      */
#define BLOCK_SIZE 32
//#define NUM_THREADS 4
//#define FOLDS 10
#define MAT_AT(mat, i, j) ((mat)->data[(i) * (mat)->cols + (j)])

typedef struct CSRMatrix CSRMatrix;
typedef struct CSR3DTensor CSR3DTensor;

typedef struct {
    int rows;
    int cols;
    double** data;
    const CSRMatrix* csr_view;  /* Vue CSR optionnelle pour les tranches non denses. */
} Matrix;

struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    double* values;
    int* col_index;
    int* row_ptr;
};

typedef struct {
    Matrix* slices;
    int num_slices;
    int rows;
    int cols;
    CSR3DTensor* csr;
    int owns_csr;
} Tensor3D;

struct CSR3DTensor {
    CSRMatrix* slices;
    int num_slices;
    int rows;
    int cols;
};

typedef struct {
    double *data;
    size_t size;
    size_t capacity;
} Array;
typedef struct{
    Matrix* A;
    Tensor3D* R;
    double f;
    int itr;
    Array exectimes;
}resultat;

typedef struct {
    Matrix U;
    double* S; 
    Matrix Vt;
} SVDResult;

typedef struct {
    int index;
    double value;
} IndexValuePair;

typedef struct {
    double score;
    int true_label;
} ScoreLabelPair;

typedef struct {
    char** filenames;
    int count;
    int capacity;
} FileList;

/* Nombre de threads pour le produit matriciel parallele */
/* Initialisee dans main(), lue par dot() et ses workers  */
extern int g_num_threads;

void check_slices(Tensor3D* X);

CSRMatrix* dense_to_csr(Matrix* dense);

CSR3DTensor* tensor_to_csr(Tensor3D* tensor);

Tensor3D* tensor_from_csr(CSR3DTensor* csr_tensor, int take_ownership);

double tensor_get_value(const Tensor3D* tensor, int slice, int row, int col);

int tensor_set_zero(Tensor3D* tensor, int slice, int row, int col);

Matrix* init_Matrix(int rows, int cols);

void free_Matrix(Matrix* matrix);

CSRMatrix* init_CSRMatrix(int rows, int cols, int nnz);

void init_CSRMatrix_in_place(CSRMatrix* matrix, int rows, int cols, int nnz);

void free_CSRMatrix(CSRMatrix* matrix);

Tensor3D* init_tensor3D(int num_slices, int rows, int cols);

void free_tensor(Tensor3D* tensor);

CSR3DTensor* init_CSR3DTensor(int num_slices, int rows, int cols, int nnz_per_slice);

void free_CSR3DTensor(CSR3DTensor* tensor);

resultat init_resultat(int rows, int cols,int rank, int num_slices, size_t array_size);

void free_resultat(resultat *res);

void array_init(Array *a, size_t initial_size);

void array_free(Array *a);

void array_append(Array *a, double value);

Matrix* copy_Matrix(Matrix* src);

void copy_matrix_data(Matrix* dest, Matrix* src);

Tensor3D* copy_Tensor3D(Tensor3D* src);

double norm(Matrix* matrix);

void normalize_vector(double* v, int n);

int count_non_zero(const Matrix* mat);

int is_nonzero_csr(CSRMatrix* X, int row, int col);

Matrix* dot(Matrix* A, Matrix* B);

Matrix* dot_csr(CSRMatrix* A, Matrix* B);

Matrix* dot_csr1(Matrix* A, CSRMatrix* B);

Matrix* prod(Matrix* A, Matrix* B);

double dot_product(double* v1, double* v2, int n);

void matrix_vector_multiply(Matrix* A, double* v, double* result);

Matrix* T_matrix(Matrix* A);

CSRMatrix* T_csr(CSRMatrix* A);

Matrix* get_matrix(int n, int m, unsigned int seed);

Matrix* eigsh(Matrix* A, int k, double** eigenvalues);

Matrix* eye(int n);

Matrix* multiply_by_scalar(Matrix* matrix, double scalar);

Matrix* add_matrices(Matrix* A, Matrix* B);

Matrix* soustraction(Matrix* A, Matrix* B);
    
Matrix* Inverse(Matrix* mat);

Matrix* MatrixAugmentWithIdentity(Matrix* mat);

void FreeSVDResult(SVDResult* svd);

double LocalHypot(double a, double b);

Matrix* solve(Matrix* A, Matrix* B);

Matrix* reshape(double* vector, int n);

double* kron(double *vec1, int len1, double *vec2, int len2, int *resultLen);

void save_matrix_to_file(Matrix* matrix, const char* filename);

int count_zeros(double* vec, int size, double tolerance);

Matrix* load_matrix_from_file(const char* filename);

int count_zero_elements(Matrix* matrix);

Matrix* create_scaled_identity(int size, double scalar);

void log_info(int itr, double fit, double fitchange, double secs);

void print_vector(double* vec, int size);

void printMatrix(Matrix* matrix);

void print_tensor3D(Tensor3D* tensor);  

void print_CSRMatrix(CSRMatrix* matrix);

int count_nonzero_elements(Tensor3D* tensor);

void print_nonzero_count(Tensor3D* tensor);

#endif // UTILES_H
