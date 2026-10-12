# TC5061 - Experimentación y Seguimiento con MLflow

Este proyecto implementa la migración de un modelo base de regresión desarrollado en Jupyter Notebook a un script de entrenamiento modular y parametrizado en Python, instrumentado con MLflow Tracking. El problema abordado es la predicción del rendimiento de arroz (Paddy Yield) utilizando un ensamble de árboles RandomForestRegressor sobre datos agronómicos.

Repositorio: https://github.com/oscartrix/tec-mlops-2026.git

## Estructura del Proyecto

```text
├── data/
│   └── paddy+dataset.zip          # Dataset en formato comprimido
├── notebooks/
│   └── eda_paddy.ipynb            # Notebook conservado exclusivamente para EDA
├── src/
│   └── train.py                   # Script de entrenamiento y tracking en MLflow
├── reports/
│   ├── figures/                   # Capturas de MLflow y artefactos de evaluación
│   └── TC5061_Experimentacion_MLFlow_ValenzuelaOscar.pdf
├── generate_report.py             # Script generador del reporte académico
├── run_experiments.sh             # Script para ejecutar la batería de 5 corridas
├── requirements.txt               # Dependencias fijadas del proyecto
├── .gitignore
└── README.md
```

## Requisitos Previos

- Python 3.12 o superior. La versión fijada de numpy (2.5.3) no se instala en versiones anteriores.
- Servidor MLflow local o base de datos SQLite para registro de experimentos.

## Instalación y Configuración

1. Clonar el repositorio:
```bash
git clone https://github.com/oscartrix/tec-mlops-2026.git
cd tec-mlops-2026
```

2. Crear y activar el entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Iniciar el Servidor de MLflow

Para visualizar la interfaz interactiva y registrar artefactos localmente:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

La interfaz queda accesible en http://127.0.0.1:5000.

## Ejecución del Script de Entrenamiento

El script `src/train.py` cuenta con valores por defecto funcionales y puede ejecutarse sin argumentos adicionales:

```bash
python src/train.py
```

### Argumentos de Línea de Comandos

- `--n-estimators`: Número de árboles en el ensamble (entero, por defecto: 100).
- `--max-depth`: Profundidad máxima del árbol (entero o None, por defecto: None).
- `--min-samples-split`: Mínimo de muestras requeridas para dividir un nodo (entero, por defecto: 2).
- `--min-samples-leaf`: Mínimo de muestras por nodo hoja (entero, por defecto: 1).
- `--random-state`: Semilla para el particionado de datos y el modelo (entero, por defecto: 42).
- `--test-size`: Proporción del conjunto de prueba (flotante, por defecto: 0.2).
- `--data-path`: Ruta al dataset zip o csv (por defecto: data/paddy+dataset.zip).
- `--experiment-name`: Nombre del experimento en MLflow (por defecto: Paddy_Yield_Random_Forest).
- `--run-name`: Nombre descriptivo de la corrida en MLflow (opcional).
- `--tracking-uri`: URI de tracking de MLflow (opcional). Si no se indica, el script usa la variable `MLFLOW_TRACKING_URI`; si no existe, usa el servidor en http://127.0.0.1:5000 cuando está activo, y si no lo está, registra en `sqlite:///mlflow.db`.

Ejemplo de ejecución con hiperparámetros personalizados:

```bash
python src/train.py \
  --n-estimators 100 \
  --max-depth 5 \
  --min-samples-split 2 \
  --min-samples-leaf 1 \
  --random-state 42 \
  --run-name "shallow_depth_5"
```

## Ejecución por Lotes de Experimentos

Para replicar las cinco corridas comparativas de forma consecutiva:

```bash
bash run_experiments.sh
```

## Resultados Comparativos

Las cinco corridas registradas bajo el experimento `Paddy_Yield_Random_Forest` produjeron las siguientes métricas sobre el conjunto de prueba independiente:

| Corrida | max_depth | n_estimators | min_samples_split | min_samples_leaf | RMSE (kg) | MAE (kg) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| shallow_depth_5 (Mejor) | 5 | 100 | 2 | 1 | 910.49 | 654.73 | 0.9903 |
| regularized_split10_leaf4 | None | 200 | 10 | 4 | 931.82 | 673.95 | 0.9899 |
| intermediate_depth_10_leaf_2 | 10 | 150 | 5 | 2 | 952.39 | 685.91 | 0.9894 |
| deep_trees_depth_15_n250 | 15 | 250 | 2 | 1 | 965.75 | 693.32 | 0.9891 |
| baseline_default | None | 100 | 2 | 1 | 968.59 | 695.62 | 0.9890 |

## Reproducibilidad

La reproducibilidad de los resultados está garantizada mediante:
1. Fijación explícita de `random_state=42` en la partición `train_test_split` y en la inicialización de `RandomForestRegressor`.
2. Uso de un único hilo de cómputo (`n_jobs=1`) para eliminar discrepancias de redondeo en operaciones de punto flotante entre distintos núcleos de CPU.
3. Declaración de versiones fijas en `requirements.txt`.
