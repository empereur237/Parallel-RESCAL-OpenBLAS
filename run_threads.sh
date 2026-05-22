#!/usr/bin/env bash
set -euo pipefail

# Lance RESCAL-ALS avec plusieurs nombres de threads.
# Par defaut : 2, 4, 6, 8 et 10 threads.
# Par defaut, FORCE_RERUN=1 force le recalcul meme si le CSV contient
# deja une ligne pour la meme configuration.
# Exemple pour changer la liste :
#   THREADS="1 2 4 8" ./run_threads.sh
# Exemple pour reprendre sans recalculer les lignes deja presentes :
#   FORCE_RERUN=0 ./run_threads.sh

THREADS="${THREADS:-2 4 6 8 10}"
FORCE_RERUN="${FORCE_RERUN:-1}"
RESULTS_FILE="${RESULTS_FILE:-rescal_results.csv}"
LOG_DIR="${LOG_DIR:-logs}"
EXECUTABLE="${EXECUTABLE:-./rescal}"

mkdir -p "$LOG_DIR"

echo "[Build] Compilation du programme"
make all

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERREUR : executable introuvable ou non executable : $EXECUTABLE" >&2
    exit 1
fi

echo "[Info] Resultats agreges dans : $RESULTS_FILE"
echo "[Info] Logs detailles dans     : $LOG_DIR/"
echo "[Info] Threads testes          : $THREADS"
echo "[Info] Recalcul force          : $FORCE_RERUN"
echo

for threads in $THREADS; do
    if ! [[ "$threads" =~ ^[0-9]+$ ]] || (( threads < 1 )); then
        echo "ERREUR : nombre de threads invalide : $threads" >&2
        exit 1
    fi

    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="$LOG_DIR/rescal_threads_${threads}_${timestamp}.log"

    echo "===================================================="
    echo "[Run] Execution avec ${threads} thread(s)"
    echo "===================================================="
    if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" || "$FORCE_RERUN" == "TRUE" ]]; then
        "$EXECUTABLE" "$threads" --force-rerun 2>&1 | tee "$log_file"
    else
        "$EXECUTABLE" "$threads" 2>&1 | tee "$log_file"
    fi
    echo "[Done] threads=${threads} | log=${log_file}"
    echo
done

echo "===================================================="
echo "[Fin] Toutes les executions sont terminees."
echo "[Fin] Consultez $RESULTS_FILE pour comparer les temps par thread."
echo "===================================================="
