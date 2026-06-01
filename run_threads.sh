#!/usr/bin/env bash
set -euo pipefail

# Lance RESCAL-ALS dataset par dataset avec plusieurs nombres de threads.
# Pour chaque dataset, toutes les valeurs de THREADS sont terminees avant
# de passer au dataset suivant.
# Par defaut, FORCE_RERUN=0 reprend la campagne en sautant les lignes
# deja presentes dans le CSV.
# Exemple pour changer la liste des datasets :
#   DATASETS="kinships umls nations" ./run_threads.sh
# Exemple pour changer la liste :
#   THREADS="1 2 4 8" ./run_threads.sh
# Exemple pour tout recalculer meme si les lignes existent deja :
#   FORCE_RERUN=1 ./run_threads.sh

THREADS="${THREADS:-1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64}"
DATASETS="${DATASETS:-kinships umls nations codex_small codex_medium DBpedia50}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RESULTS_FILE="${RESULTS_FILE:-rescal_results.csv}"
LOG_DIR="${LOG_DIR:-logs}"
EXECUTABLE="${EXECUTABLE:-./rescal}"
FOLD_LOGS="${FOLD_LOGS:-10}"

mkdir -p "$LOG_DIR"

echo "[Build] Compilation du programme"
make all

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERREUR : executable introuvable ou non executable : $EXECUTABLE" >&2
    exit 1
fi

if ! [[ "$FOLD_LOGS" =~ ^[0-9]+$ ]] || (( FOLD_LOGS < 1 )); then
    echo "ERREUR : FOLD_LOGS doit etre un entier >= 1" >&2
    exit 1
fi

echo "[Info] Resultats agreges dans : $RESULTS_FILE"
echo "[Info] Logs detailles dans     : $LOG_DIR/"
echo "[Info] Logs folds par thread   : $FOLD_LOGS"
echo "[Info] Datasets testes         : $DATASETS"
echo "[Info] Threads testes          : $THREADS"
echo "[Info] Recalcul force          : $FORCE_RERUN"
echo

if command -v stdbuf >/dev/null 2>&1; then
    STDBUF_CMD=(stdbuf -oL -eL)
else
    STDBUF_CMD=()
    echo "[Warn] stdbuf introuvable : l'affichage peut etre moins progressif."
fi

for dataset in $DATASETS; do
    dataset_label="${dataset//[^A-Za-z0-9_.-]/_}"

    echo "===================================================="
    echo "[Dataset] Debut du traitement complet : ${dataset}"
    echo "===================================================="

    for threads in $THREADS; do
        if ! [[ "$threads" =~ ^[0-9]+$ ]] || (( threads < 1 )); then
            echo "ERREUR : nombre de threads invalide : $threads" >&2
            exit 1
        fi

        timestamp="$(date +%Y%m%d_%H%M%S)"
        run_log_dir="$LOG_DIR/${dataset_label}/threads_${threads}_${timestamp}"
        fold_log_dir="$run_log_dir/folds"
        log_file="$run_log_dir/rescal_${dataset_label}_threads_${threads}.log"
        mkdir -p "$fold_log_dir"

        for ((fold = 0; fold < FOLD_LOGS; fold++)); do
            printf -v fold_log_file "%s/fold_%02d.log" "$fold_log_dir" "$fold"
            {
                printf "Dataset   : %s\n" "$dataset"
                printf "Thread(s) : %s\n" "$threads"
                printf "Fold      : %02d\n" "$fold"
                printf "Debut     : %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
                printf "====================================================\n"
            } > "$fold_log_file"
        done

        rescal_cmd=("$EXECUTABLE" "$threads" "--dataset" "$dataset")
        if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" || "$FORCE_RERUN" == "TRUE" ]]; then
            rescal_cmd+=("--force-rerun")
        fi

        echo "===================================================="
        echo "[Run] Dataset : ${dataset}"
        echo "[Run] Threads : ${threads}"
        echo "[Run] Log complet : ${log_file}"
        echo "[Run] Logs folds  : ${fold_log_dir}/fold_XX.log"
        echo "===================================================="

        "${STDBUF_CMD[@]}" "${rescal_cmd[@]}" 2>&1 | awk \
            -v full_log="$log_file" \
            -v fold_dir="$fold_log_dir" \
            -v fold_count="$FOLD_LOGS" '
        function fold_path(f) {
            return sprintf("%s/fold_%02d.log", fold_dir, f)
        }

        function start_fold_log(f, path, i) {
            if (f < 0 || f >= fold_count) {
                current_fold = -1
                return
            }

            current_fold = f
            path = fold_path(f)
            print "" >> path
            print "====================================================" >> path
            print sprintf("Fold %02d", f) >> path
            print "====================================================" >> path
            if (dataset_line != "") {
                print dataset_line >> path
            }
            if (config_line != "") {
                print config_line >> path
            }
            for (i = 1; i <= data_count; i++) {
                print data_lines[i] >> path
            }
            if (eval_line != "") {
                print eval_line >> path
            }
            fflush(path)
        }

        function write_fold_line(line, path) {
            if (current_fold < 0 || current_fold >= fold_count) {
                return
            }
            path = fold_path(current_fold)
            print line >> path
            fflush(path)
        }

        {
            print
            fflush()
            print >> full_log
            fflush(full_log)

            if ($0 ~ /^=== DATASET:/) {
                dataset_line = $0
                config_line = ""
                eval_line = ""
                data_count = 0
                in_eval = 0
                next_fold = 0
                current_fold = -1
                next
            }

            if ($0 ~ /^\[Config\]/) {
                config_line = $0
                next
            }

            if ($0 ~ /^\[Data\]/) {
                data_lines[++data_count] = $0
                next
            }

            if ($0 ~ /^\[Eval\]/) {
                eval_line = $0
                in_eval = 1
                next_fold = 0
                current_fold = -1
                next
            }

            if (in_eval && ($0 ~ /^INFO : RESCAL/ || $0 ~ /^\[Fold [0-9]+\]/)) {
                if (current_fold < 0) {
                    start_fold_log(next_fold)
                }
                write_fold_line($0)

                if ($0 ~ /^\[Fold [0-9]+\]/) {
                    fold_number = $0
                    sub(/^\[Fold /, "", fold_number)
                    sub(/\].*/, "", fold_number)
                    if (current_fold >= 0 && current_fold < fold_count) {
                        close(fold_path(current_fold))
                    }
                    current_fold = -1
                    next_fold = fold_number + 1
                }
            }
        }
        END {
            close(full_log)
        }
        '

        echo "[Done] dataset=${dataset} threads=${threads} | log complet=${log_file}"
        echo "[Done] logs folds=${fold_log_dir}/"
        if [[ -f "$RESULTS_FILE" ]]; then
            echo "[CSV] Resultats disponibles dans ${RESULTS_FILE}"
        fi
        echo
    done

    echo "[Dataset] Termine : ${dataset}"
    echo
done

echo "===================================================="
echo "[Fin] Toutes les executions sont terminees."
echo "[Fin] Consultez $RESULTS_FILE pour comparer les temps par dataset et par thread."
echo "===================================================="
