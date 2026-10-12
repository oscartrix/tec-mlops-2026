#!/usr/bin/env bash
set -e

# Script de ejecucion sistematica de experimentos con MLflow Tracking
# Entorno virtual y configuracion
PYTHON_EXEC="/home/oavm/Documentos/school/mlops/.venv/bin/python"
EXPERIMENT_NAME="Paddy_Yield_Random_Forest"

echo "============================================================"
echo "Iniciando bateria de experimentos para: ${EXPERIMENT_NAME}"
echo "============================================================"

# Corrida 1: Linea base (configuracion por defecto)
echo ""
echo ">>> Ejecutando Corrida 1: Linea Base (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)"
$PYTHON_EXEC src/train.py \
  --n-estimators 100 \
  --max-depth None \
  --min-samples-split 2 \
  --min-samples-leaf 1 \
  --random-state 42 \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-name "baseline_default"

# Corrida 2: Arboles superficiales (alta regularizacion estructural)
echo ""
echo ">>> Ejecutando Corrida 2: Arboles Superficiales (n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1)"
$PYTHON_EXEC src/train.py \
  --n-estimators 100 \
  --max-depth 5 \
  --min-samples-split 2 \
  --min-samples-leaf 1 \
  --random-state 42 \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-name "shallow_depth_5"

# Corrida 3: Profundidad intermedia con restriccion en hojas
echo ""
echo ">>> Ejecutando Corrida 3: Profundidad Intermedia (n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2)"
$PYTHON_EXEC src/train.py \
  --n-estimators 150 \
  --max-depth 10 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --random-state 42 \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-name "intermediate_depth_10_leaf_2"

# Corrida 4: Mayor capacidad y bosque mas amplio
echo ""
echo ">>> Ejecutando Corrida 4: Ensamble Amplio y Profundo (n_estimators=250, max_depth=15, min_samples_split=2, min_samples_leaf=1)"
$PYTHON_EXEC src/train.py \
  --n-estimators 250 \
  --max-depth 15 \
  --min-samples-split 2 \
  --min-samples-leaf 1 \
  --random-state 42 \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-name "deep_trees_depth_15_n250"

# Corrida 5: Regularizacion de division y hojas sin limite estricto de profundidad
echo ""
echo ">>> Ejecutando Corrida 5: Regularizacion de Nodos y Hojas (n_estimators=200, max_depth=None, min_samples_split=10, min_samples_leaf=4)"
$PYTHON_EXEC src/train.py \
  --n-estimators 200 \
  --max-depth None \
  --min-samples-split 10 \
  --min-samples-leaf 4 \
  --random-state 42 \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-name "regularized_split10_leaf4"

echo ""
echo "============================================================"
echo "Bateria de experimentos finalizada exitosamente."
echo "Visualice y compare los resultados en http://127.0.0.1:5000"
echo "============================================================"
