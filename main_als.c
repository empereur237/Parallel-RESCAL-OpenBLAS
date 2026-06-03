/*
 * Auteur      : tonlio djiogo nelson mandela (Projet RESCAL-ALS avec BLAS )
 * Date        : 2026
 * Description : chargement des tenseurs, prédiction et métriques.
 */

#include "predition.h"
#include "utiles.h"

#include <errno.h>
#include <stdbool.h>
#include <unistd.h>

#define RESULTS_FILE "rescal_results.csv"
#define FOLDS 10
#define MAX_VALIDATION_POSITIVES 10000
#define MAX_LARGE_VALIDATION_POSITIVES 2000
#define NEGATIVES_PER_POSITIVE 1
#define LARGE_TENSOR_DENSE_ENTRIES_THRESHOLD 100000000LL
#define ARRAY_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

typedef struct {
    int rank;
    int maxIter;
    double conv;
    double lambda_A;
    double lambda_R;
    double lambda_Z;
} ALSConfig;

typedef struct {
    const char* name;
    const char* path;
    ALSConfig* configs;
    int num_configs;
} DatasetConfig;

typedef struct {
    char** keys;
    int size;
    int capacity;
} ProcessedSet;

typedef struct {
    int row;
    int col;
    int slice;
    int label;
} ValidationSample;

static double wall_seconds(void) {
    return omp_get_wtime();
}

static void shuffle_validation_samples(ValidationSample* samples, int size) {
    if (!samples || size <= 1) {
        return;
    }

    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        ValidationSample tmp = samples[i];
        samples[i] = samples[j];
        samples[j] = tmp;
    }
}

static ValidationSample* collect_positive_samples(Tensor3D* tensor, int* out_count) {
    if (!tensor || !out_count) {
        return NULL;
    }

    int capacity = count_nonzero_elements(tensor);
    if (capacity <= 0) {
        *out_count = 0;
        return NULL;
    }

    ValidationSample* samples = malloc((size_t)capacity * sizeof(ValidationSample));
    if (!samples) {
        fprintf(stderr, "Erreur : allocation des echantillons positifs\n");
        *out_count = 0;
        return NULL;
    }

    int count = 0;
    if (tensor->csr) {
        for (int k = 0; k < tensor->csr->num_slices; k++) {
            CSRMatrix* slice = &tensor->csr->slices[k];
            for (int row = 0; row < slice->rows; row++) {
                for (int p = slice->row_ptr[row]; p < slice->row_ptr[row + 1]; p++) {
                    if (fabs(slice->values[p]) > EPSILON && count < capacity) {
                        samples[count++] = (ValidationSample){
                            row, slice->col_index[p], k, 1
                        };
                    }
                }
            }
        }
    } else {
        for (int k = 0; k < tensor->num_slices; k++) {
            for (int row = 0; row < tensor->rows; row++) {
                for (int col = 0; col < tensor->cols; col++) {
                    if (fabs(tensor->slices[k].data[row][col]) > EPSILON && count < capacity) {
                        samples[count++] = (ValidationSample){row, col, k, 1};
                    }
                }
            }
        }
    }

    *out_count = count;
    return samples;
}

static ValidationSample random_negative_sample(Tensor3D* tensor) {
    ValidationSample sample = {0, 0, 0, 0};

    for (int attempts = 0; attempts < 10000; attempts++) {
        int row = rand() % tensor->rows;
        int col = rand() % tensor->cols;
        int slice = rand() % tensor->num_slices;

        if (fabs(tensor_get_value(tensor, slice, row, col)) <= EPSILON) {
            sample.row = row;
            sample.col = col;
            sample.slice = slice;
            return sample;
        }
    }

    return sample;
}

static double score_rescal_entry(Matrix* A, Tensor3D* R, int slice, int row, int col) {
    if (!A || !R || slice < 0 || slice >= R->num_slices ||
        row < 0 || row >= A->rows || col < 0 || col >= A->rows) {
        return 0.0;
    }

    Matrix* Rk = &R->slices[slice];
    double score = 0.0;

    for (int i = 0; i < A->cols; i++) {
        double left = A->data[row][i];
        if (fabs(left) <= EPSILON) {
            continue;
        }

        for (int j = 0; j < A->cols; j++) {
            score += left * Rk->data[i][j] * A->data[col][j];
        }
    }

    return score;
}

static double run_sampled_fold_auc(Tensor3D* T, Tensor3D* P,
                                   ValidationSample* positives, int positive_count,
                                   int rank, int maxIter, double conv,
                                   double lambda_A, double lambda_R, double lambda_Z) {
    if (!T || !positives || positive_count <= 0) {
        return 0.0;
    }

    Tensor3D* T_train = copy_Tensor3D(T);
    if (!T_train) {
        fprintf(stderr, "Erreur : copie du tenseur d'entrainement impossible\n");
        return 0.0;
    }

    for (int i = 0; i < positive_count; i++) {
        tensor_set_zero(T_train, positives[i].slice, positives[i].row, positives[i].col);
    }

    resultat trained_model = rescal_als(T_train, rank, "random",
                                        maxIter, conv,
                                        lambda_A, lambda_R, lambda_Z,
                                        P, 0);
    free_tensor(T_train);

    if (!trained_model.A || !trained_model.R) {
        free_resultat(&trained_model);
        return 0.0;
    }

    int total_samples = positive_count * (1 + NEGATIVES_PER_POSITIVE);
    double* y_true = malloc((size_t)total_samples * sizeof(double));
    double* y_pred = malloc((size_t)total_samples * sizeof(double));
    if (!y_true || !y_pred) {
        fprintf(stderr, "Erreur : allocation des scores de validation\n");
        free(y_true);
        free(y_pred);
        free_resultat(&trained_model);
        return 0.0;
    }

    int pos = 0;
    for (int i = 0; i < positive_count; i++) {
        y_true[pos] = 1.0;
        y_pred[pos] = score_rescal_entry(trained_model.A, trained_model.R,
                                         positives[i].slice,
                                         positives[i].row,
                                         positives[i].col);
        pos++;

        for (int n = 0; n < NEGATIVES_PER_POSITIVE; n++) {
            ValidationSample neg = random_negative_sample(T);
            y_true[pos] = 0.0;
            y_pred[pos] = score_rescal_entry(trained_model.A, trained_model.R,
                                             neg.slice, neg.row, neg.col);
            pos++;
        }
    }

    double auc = compute_auc_pr(y_true, y_pred, total_samples);

    free(y_true);
    free(y_pred);
    free_resultat(&trained_model);
    return auc;
}

static void processed_set_init(ProcessedSet* set) {
    set->size = 0;
    set->capacity = 16;
    set->keys = malloc((size_t)set->capacity * sizeof(char*));
    if (!set->keys) {
        fprintf(stderr, "Erreur : allocation de l'ensemble de reprise\n");
        exit(EXIT_FAILURE);
    }
}

static void processed_set_free(ProcessedSet* set) {
    for (int i = 0; i < set->size; i++) {
        free(set->keys[i]);
    }
    free(set->keys);
    set->keys = NULL;
    set->size = 0;
    set->capacity = 0;
}

static bool processed_set_contains(const ProcessedSet* set, const char* key) {
    for (int i = 0; i < set->size; i++) {
        if (strcmp(set->keys[i], key) == 0) {
            return true;
        }
    }
    return false;
}

static void processed_set_add(ProcessedSet* set, const char* key) {
    if (processed_set_contains(set, key)) {
        return;
    }

    if (set->size == set->capacity) {
        int new_capacity = set->capacity * 2;
        char** new_keys = realloc(set->keys, (size_t)new_capacity * sizeof(char*));
        if (!new_keys) {
            fprintf(stderr, "Erreur : reallocation de l'ensemble de reprise\n");
            exit(EXIT_FAILURE);
        }
        set->keys = new_keys;
        set->capacity = new_capacity;
    }

    size_t len = strlen(key) + 1;
    set->keys[set->size] = malloc(len);
    if (!set->keys[set->size]) {
        fprintf(stderr, "Erreur : allocation d'une cle de reprise\n");
        exit(EXIT_FAILURE);
    }
    memcpy(set->keys[set->size], key, len);
    set->size++;
}

static void make_run_key(char* buffer, size_t size, const char* dataset,
                         int rank, int maxIter, double conv,
                         double lambda_A, double lambda_R, int threads) {
    snprintf(buffer, size, "%s;%d;%d;%.12g;%.12g;%.12g;%d",
             dataset, rank, maxIter, conv, lambda_A, lambda_R, threads);
}

static void load_processed_runs(const char* filename, ProcessedSet* set) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return;
    }

    char line[2048];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return;
    }

    while (fgets(line, sizeof(line), file)) {
        char dataset[256];
        int rank = 0;
        int maxIter = 0;
        int threads = 0;
        double conv = 0.0;
        double lambda_A = 0.0;
        double lambda_R = 0.0;

        char* fields[12] = {0};
        int field_count = 0;
        char* token = strtok(line, ";\n\r");
        while (token && field_count < 12) {
            fields[field_count++] = token;
            token = strtok(NULL, ";\n\r");
        }

        if (field_count >= 12) {
            snprintf(dataset, sizeof(dataset), "%s", fields[0]);
            rank = atoi(fields[1]);
            maxIter = atoi(fields[2]);
            conv = strtod(fields[3], NULL);
            lambda_A = strtod(fields[4], NULL);
            lambda_R = strtod(fields[5], NULL);
            threads = atoi(fields[11]);

            char key[512];
            make_run_key(key, sizeof(key), dataset, rank, maxIter, conv,
                         lambda_A, lambda_R, threads);
            processed_set_add(set, key);
        }
    }

    fclose(file);
}

static FILE* open_results_file(const char* filename) {
    bool exists = (access(filename, F_OK) == 0);
    FILE* file = fopen(filename, "a");
    if (!file) {
        fprintf(stderr, "Erreur : impossible d'ouvrir %s (%s)\n", filename, strerror(errno));
        return NULL;
    }

    if (!exists) {
        fprintf(file, "dataset;rank;maxIter;conv;lambda_A;lambda_R;nnz;"
                      "import_time_s;auc_pr_mean;auc_pr_std;total_time_s;threads\n");
        fflush(file);
    }

    return file;
}

int main(void) {
    srand((unsigned int)time(NULL));

    ALSConfig dbpedia50_configs[] = {
        { 150, 20, 1e-5, 0.5, 0.5, 5.0 },
    };

    ALSConfig codex_small_configs[] = {
        { 100, 30, 1e-5, 0.5, 0.5, 5.0 },
    };

    ALSConfig nations_configs[] = {
        { 90, 100, 1e-4, 5.0, 5.0, 5.0 },
    };

    ALSConfig umls_configs[] = {
        { 90, 100, 1e-4, 5.0, 5.0, 5.0 },
    };

    ALSConfig kinships_configs[] = {
        { 90, 100, 1e-4, 5.0, 5.0, 5.0 },
    };

    ALSConfig codex_medium_configs[] = {
        { 150, 20, 1e-5, 0.2, 0.2, 5.0 },
    };

    ALSConfig fb15k237_configs[] = {
        { 200, 20, 1e-6, 0.1, 0.1, 5.0 },
    };

    DatasetConfig datasets[] = {
        { "kinships",      "./kinships",          kinships_configs,      ARRAY_LEN(kinships_configs) },
        { "umls",          "./umls",              umls_configs,          ARRAY_LEN(umls_configs) },
        { "nations",       "./nations",           nations_configs,       ARRAY_LEN(nations_configs) },
        { "fb15k237",      "./fb15k237_csr",      fb15k237_configs,      ARRAY_LEN(fb15k237_configs) },
        { "codex_small",   "./codex_small_csr",   codex_small_configs,   ARRAY_LEN(codex_small_configs) },
        { "codex_medium",  "./codex_medium_csr",  codex_medium_configs,  ARRAY_LEN(codex_medium_configs) },
        { "DBpedia50",     "./DBpedia50_csr",     dbpedia50_configs,     ARRAY_LEN(dbpedia50_configs) },
    };
    int num_datasets = ARRAY_LEN(datasets);

    int thread_values[] = {
        1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
    };
    int num_thread_values = ARRAY_LEN(thread_values);

    ProcessedSet processed;
    processed_set_init(&processed);
    load_processed_runs(RESULTS_FILE, &processed);

    FILE* results_file = open_results_file(RESULTS_FILE);
    if (!results_file) {
        processed_set_free(&processed);
        return EXIT_FAILURE;
    }

    int total_combinations = 0;
    for (int d = 0; d < num_datasets; d++) {
        total_combinations += datasets[d].num_configs * num_thread_values;
    }

    int executed = 0;
    int attr = 0;

    for (int d = 0; d < num_datasets; d++) {
        DatasetConfig dataset = datasets[d];

        printf("====================================================\n");
        printf("DATASET: %s\n", dataset.name);
        printf("====================================================\n");

        double import_start = wall_seconds();
        Tensor3D* T = load_tensor_from_directory(dataset.path);
        double import_time = wall_seconds() - import_start;

        if (!T) {
            fprintf(stderr, "Erreur : impossible de charger le tenseur depuis %s\n", dataset.path);
            continue;
        }

        int e = T->rows;
        int nnz = count_nonzero_elements(T);
        long long dense_entries = (long long)T->rows * T->cols * T->num_slices;
        printf("[Data]   nnz:%d | import_time:%.2f s\n", nnz, import_time);

        int positive_count = 0;
        ValidationSample* positives = collect_positive_samples(T, &positive_count);
        if (!positives || positive_count <= 0) {
            fprintf(stderr, "Erreur : aucun echantillon positif pour la validation\n");
            free(positives);
            free_tensor(T);
            continue;
        }

        shuffle_validation_samples(positives, positive_count);
        int validation_cap = (dense_entries > LARGE_TENSOR_DENSE_ENTRIES_THRESHOLD)
            ? MAX_LARGE_VALIDATION_POSITIVES
            : MAX_VALIDATION_POSITIVES;
        if (positive_count > validation_cap) {
            positive_count = validation_cap;
        }

        int requested_folds = FOLDS;
        int active_folds = (positive_count < requested_folds) ? positive_count : requested_folds;
        int fold_size = positive_count / active_folds;
        printf("[Eval]   positifs:%d | neg/pos:%d | folds:%d | mode:%s\n",
               positive_count, NEGATIVES_PER_POSITIVE, active_folds,
               (dense_entries > LARGE_TENSOR_DENSE_ENTRIES_THRESHOLD)
                   ? "cross-validation CSR echantillonnee"
                   : "cross-validation");

        for (int t = 0; t < num_thread_values; t++) {
            int threads = thread_values[t];
            omp_set_num_threads(threads);
            openblas_set_num_threads(threads);

            for (int c = 0; c < dataset.num_configs; c++) {
                ALSConfig config = dataset.configs[c];
                char run_key[512];
                make_run_key(run_key, sizeof(run_key), dataset.name, config.rank,
                             config.maxIter, config.conv,
                             config.lambda_A, config.lambda_R, threads);

                if (processed_set_contains(&processed, run_key)) {
                    printf("SKIP: [%s] rank=%d threads=%d - deja calcule\n",
                           dataset.name, config.rank, threads);
                    continue;
                }

                printf("=== RUN: %s | threads:%d | rank:%d | lambda_A:%.3g | lambda_R:%.3g ===\n",
                       dataset.name, threads, config.rank, config.lambda_A, config.lambda_R);
                printf("[Config] maxIter:%d | conv:%.1e | OpenBLAS threads:%d\n",
                       config.maxIter, config.conv, threads);

                Tensor3D* P = NULL;
                if (attr > 0) {
                    P = create_random_binary_tensor(attr, e, config.rank);
                }

                double* AUC_test = calloc(FOLDS, sizeof(double));
                if (!AUC_test) {
                    fprintf(stderr, "Erreur : allocation des resultats de validation\n");
                    if (P) free_tensor(P);
                    continue;
                }

                double run_start = wall_seconds();

                for (int f = 0; f < active_folds; f++) {
                    int start = f * fold_size;
                    int current_fold_size = (f == active_folds - 1)
                        ? positive_count - start
                        : fold_size;

                    AUC_test[f] = run_sampled_fold_auc(
                        T, P, positives + start, current_fold_size,
                        config.rank, config.maxIter, config.conv,
                        config.lambda_A, config.lambda_R, config.lambda_Z
                    );
                    printf("[Fold %d] AUC-PR: %.6f\n", f, AUC_test[f]);
                }

                double total_time = wall_seconds() - run_start;
                double test_mean = mean(AUC_test, active_folds);
                double test_std = stddev(AUC_test, active_folds);

                printf("[Result] AUC-PR mean:%.6f | std:%.6f | time:%.2f s\n",
                       test_mean, test_std, total_time);

                fprintf(results_file,
                        "%s;%d;%d;%.6f;%.6f;%.6f;%d;%.6f;%.6f;%.6f;%.6f;%d\n",
                        dataset.name, config.rank, config.maxIter, config.conv,
                        config.lambda_A, config.lambda_R, nnz, import_time,
                        test_mean, test_std, total_time, threads);
                fflush(results_file);
                processed_set_add(&processed, run_key);
                executed++;

                printf("[CSV]    Ligne écrite dans %s\n", RESULTS_FILE);
                printf("=== FIN: %s rank:%d threads:%d ===\n",
                       dataset.name, config.rank, threads);

                free(AUC_test);
                if (P) free_tensor(P);
            }
        }

        free(positives);
        free_tensor(T);
    }

    printf("====================================================\n");
    printf("RÉSUMÉ FINAL\n");
    printf("====================================================\n");
    printf("Combinaisons exécutées : %d / %d\n", executed, total_combinations);
    printf("Résultats sauvegardés  : %s\n", RESULTS_FILE);
    printf("====================================================\n");

    fclose(results_file);
    processed_set_free(&processed);
    return EXIT_SUCCESS;
}
