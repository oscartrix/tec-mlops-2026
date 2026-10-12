#!/usr/bin/env bash
# Ejecuta las cinco corridas comparativas del experimento.
# Usa el python del entorno activo; puede cambiarse con la variable PYTHON.
set -e

PYTHON="${PYTHON:-python}"
EXPERIMENT_NAME="Paddy_Yield_Random_Forest"

run() {
  local name="$1"; shift
  echo ">>> Corrida: ${name}"
  "$PYTHON" src/train.py --experiment-name "$EXPERIMENT_NAME" --random-state 42 --run-name "$name" "$@"
}

run baseline_default             --n-estimators 100 --max-depth None --min-samples-split 2  --min-samples-leaf 1
run shallow_depth_5              --n-estimators 100 --max-depth 5    --min-samples-split 2  --min-samples-leaf 1
run intermediate_depth_10_leaf_2 --n-estimators 150 --max-depth 10   --min-samples-split 5  --min-samples-leaf 2
run deep_trees_depth_15_n250     --n-estimators 250 --max-depth 15   --min-samples-split 2  --min-samples-leaf 1
run regularized_split10_leaf4    --n-estimators 200 --max-depth None --min-samples-split 10 --min-samples-leaf 4

echo "Corridas finalizadas. Compare los resultados en la interfaz de MLflow."
