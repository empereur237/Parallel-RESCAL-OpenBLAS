/*
 * Auteur      : Projet RESCAL-ALS
 * Date        : 2026
 * Description : exécution séquentielle des expériences RESCAL-ALS.
 */

#include "predition.h"
#include "utiles.h"

#include <errno.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>

#define RESULTS_FILE "rescal_results.csv"
#define FOLDS 10

typedef struct {
    const char* name;
    const char* path;
} DatasetConfig;

typedef struct {
    int rank;
    int maxIter;
    double conv;
    double lambda_A;
    double lambda_R;
    double lambda_Z;
} ALSConfig;

typedef struct {
    char** keys;
    int size;
    int capacity;
} ProcessedSet;

static double elapsed_seconds(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
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
                         int rank, double lambda_A, double lambda_R, int threads) {
    snprintf(buffer, size, "%s;%d;%.6f;%.6f;%d",
             dataset, rank, lambda_A, lambda_R, threads);
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
        double lambda_A = 0.0;
        double lambda_R = 0.0;
        int threads = 1;

        char* fields[12] = {0};
        int field_count = 0;
        char* token = strtok(line, ";\n\r");
        while (token && field_count < 12) {
            fields[field_count++] = token;
            token = strtok(NULL, ";\n\r");
        }

        if (field_count >= 6) {
            snprintf(dataset, sizeof(dataset), "%s", fields[0]);
            rank = atoi(fields[1]);
            lambda_A = strtod(fields[4], NULL);
            lambda_R = strtod(fields[5], NULL);
            if (field_count >= 12) {
                threads = atoi(fields[11]);
                if (threads < 1) {
                    threads = 1;
                }
            }

            char key[512];
            make_run_key(key, sizeof(key), dataset, rank, lambda_A, lambda_R, threads);
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

static bool is_force_value(const char* value) {
    return value &&
           (strcmp(value, "1") == 0 ||
            strcmp(value, "true") == 0 ||
            strcmp(value, "TRUE") == 0 ||
            strcmp(value, "yes") == 0 ||
            strcmp(value, "YES") == 0);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    /* Nombre de threads : argument optionnel */
    /* Usage : ./rescal [num_threads] [--force-rerun] */
    /* Defaut : 1 thread (sequentiel)          */
    int max_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (max_threads < 1) {
        max_threads = 1;
    }

    bool force_rerun = is_force_value(getenv("RESCAL_FORCE_RERUN"));
    bool threads_set = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--force-rerun") == 0 ||
            strcmp(argv[i], "--no-skip") == 0) {
            force_rerun = true;
            continue;
        }

        if (argv[i][0] == '-') {
            fprintf(stderr, "ERREUR : option inconnue : %s\n", argv[i]);
            fprintf(stderr, "Usage : %s [num_threads] [--force-rerun]\n", argv[0]);
            return EXIT_FAILURE;
        }

        if (threads_set) {
            fprintf(stderr, "ERREUR : nombre de threads donne plusieurs fois\n");
            fprintf(stderr, "Usage : %s [num_threads] [--force-rerun]\n", argv[0]);
            return EXIT_FAILURE;
        }

        char* endptr = NULL;
        long parsed_threads = strtol(argv[i], &endptr, 10);
        if (!endptr || *endptr != '\0' || parsed_threads < 1) {
            fprintf(stderr, "ERREUR : num_threads doit etre un entier >= 1\n");
            return EXIT_FAILURE;
        }

        g_num_threads = (int)parsed_threads;
        threads_set = true;
    }

    if (!threads_set) {
        g_num_threads = 1;
    }

    if (g_num_threads > max_threads) {
        fprintf(stderr,
            "AVERTISSEMENT : %d threads demandes, "
            "machine limitee a %d. Valeur reduite.\n",
            g_num_threads, max_threads);
        g_num_threads = max_threads;
    }

    printf("[Threads] Utilisation de %d thread(s) sur %d disponibles\n",
           g_num_threads, max_threads);
    if (force_rerun) {
        printf("[Reprise] Mode force active : les combinaisons deja presentes seront recalculees\n");
    }

    DatasetConfig datasets[] = {
        { "kinships", "./kinships" },
        { "umls",     "./umls"     },
        { "nations",  "./nations"  },
        /* { "last_fm", "./last_fm" }, */
    };
    int num_datasets = (int)(sizeof(datasets) / sizeof(datasets[0]));

    ALSConfig configs[] = {
        { 50,  100, 1e-4, 5.0,  5.0,  5.0 },
        { 90,  100, 1e-4, 5.0,  5.0,  5.0 },
        { 100, 100, 1e-4, 10.0, 10.0, 5.0 },
        /* Ajouter d'autres configurations ici. */
    };
    int num_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    ProcessedSet processed;
    processed_set_init(&processed);
    load_processed_runs(RESULTS_FILE, &processed);

    FILE* results_file = open_results_file(RESULTS_FILE);
    if (!results_file) {
        processed_set_free(&processed);
        return EXIT_FAILURE;
    }

    int total_combinations = num_datasets * num_configs;
    int executed = 0;
    int attr = 0;

    for (int d = 0; d < num_datasets; d++) {
        for (int c = 0; c < num_configs; c++) {
            DatasetConfig dataset = datasets[d];
            ALSConfig config = configs[c];
            char run_key[512];
            make_run_key(run_key, sizeof(run_key), dataset.name, config.rank,
                         config.lambda_A, config.lambda_R, g_num_threads);

            if (!force_rerun && processed_set_contains(&processed, run_key)) {
                printf("SKIP: [%s] rank=%d threads=%d -- deja calcule\n",
                       dataset.name, config.rank, g_num_threads);
                continue;
            }

            printf("=== DATASET: %s | rank:%d | lambda_A:%.1f | lambda_R:%.1f ===\n",
                   dataset.name, config.rank, config.lambda_A, config.lambda_R);
            printf("[Config] maxIter:%d | conv:%.1e | threads:%d\n",
                   config.maxIter, config.conv, g_num_threads);

            struct timeval import_start, import_end;
            gettimeofday(&import_start, NULL);
            Tensor3D* T = load_tensor_from_directory(dataset.path);
            gettimeofday(&import_end, NULL);
            double import_time = elapsed_seconds(import_start, import_end);

            if (!T) {
                fprintf(stderr, "Erreur : impossible de charger le tenseur depuis %s\n", dataset.path);
                continue;
            }

            int e = T->rows;
            int k = T->num_slices;
            int nnz = count_nonzero_elements(T);
            int tensor_size = e * e * k;
            printf("[Data]   nnz:%d | import_time:%.2f s\n", nnz, import_time);

            Tensor3D* P = NULL;
            if (attr > 0) {
                P = create_random_binary_tensor(attr, e, config.rank);
            }

            int* IDX = malloc((size_t)tensor_size * sizeof(int));
            double* AUC_test = calloc(FOLDS, sizeof(double));
            if (!IDX || !AUC_test) {
                fprintf(stderr, "Erreur : allocation des donnees de validation\n");
                free(IDX);
                free(AUC_test);
                free_tensor(T);
                if (P) free_tensor(P);
                continue;
            }

            for (int i = 0; i < tensor_size; i++) {
                IDX[i] = i;
            }
            shuffle(IDX, tensor_size);

            int fold_size = tensor_size / FOLDS;
            struct timeval run_start, run_end;
            gettimeofday(&run_start, NULL);

            for (int f = 0; f < FOLDS; f++) {
                int* idx_test = IDX + f * fold_size;
                Tensor3D* predicted_tensor = innerfold(
                    T, P, idx_test, fold_size,
                    config.rank, config.maxIter, config.conv,
                    config.lambda_A, config.lambda_R, config.lambda_Z
                );

                if (!predicted_tensor) {
                    fprintf(stderr, "Erreur : prediction impossible au fold %d\n", f);
                    AUC_test[f] = 0.0;
                    continue;
                }

                AUC_test[f] = calculate_auc_pr(T, idx_test, fold_size, predicted_tensor);
                printf("[Fold %d] AUC-PR: %.6f\n", f, AUC_test[f]);
                free_tensor(predicted_tensor);
            }

            gettimeofday(&run_end, NULL);
            double total_time = elapsed_seconds(run_start, run_end);
            double test_mean = mean(AUC_test, FOLDS);
            double test_std = stddev(AUC_test, FOLDS);

            printf("[Result] AUC-PR mean:%.6f | std:%.6f | time:%.2f s\n",
                   test_mean, test_std, total_time);

            fprintf(results_file,
                    "%s;%d;%d;%.6f;%.6f;%.6f;%d;%.6f;%.6f;%.6f;%.6f;%d\n",
                    dataset.name, config.rank, config.maxIter, config.conv,
                    config.lambda_A, config.lambda_R, nnz, import_time,
                    test_mean, test_std, total_time, g_num_threads);
            fflush(results_file);
            processed_set_add(&processed, run_key);
            executed++;

            printf("[CSV]    Ligne écrite dans %s\n", RESULTS_FILE);
            printf("=== FIN: %s rank:%d ===\n", dataset.name, config.rank);

            free(IDX);
            free(AUC_test);
            free_tensor(T);
            if (P) free_tensor(P);
        }
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
