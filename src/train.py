"""Entrenamiento de un RandomForestRegressor para estimar el rendimiento de arroz
(Paddy yield) con registro de parámetros, métricas, artefactos y modelo en MLflow.

Ejemplo de uso:
    python src/train.py --n-estimators 100 --max-depth 5 --random-state 42
"""

import argparse
import json
import os
import socket
import tempfile
import zipfile
from typing import Dict, Tuple
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

TARGET_COLUMN = "Paddy yield(in Kg)"
DEFAULT_SERVER_URI = "http://127.0.0.1:5000"
LOCAL_TRACKING_URI = "sqlite:///mlflow.db"


def parse_args() -> argparse.Namespace:
    """Define los argumentos de línea de comandos y sus valores por defecto."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de RandomForestRegressor para el rendimiento de arroz con seguimiento en MLflow."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Número de árboles en el ensamble (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=lambda v: None if v in ("None", "none", "null", "") else int(v),
        default=None,
        help="Profundidad máxima de cada árbol; None permite crecer sin límite (default: None).",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Número mínimo de muestras para dividir un nodo interno (default: 2).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Número mínimo de muestras en cada hoja (default: 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla usada en la partición de datos y en el modelo (default: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de datos reservada para prueba (default: 0.2).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/paddy+dataset.zip",
        help="Ruta al dataset en formato zip o csv (default: data/paddy+dataset.zip).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Paddy_Yield_Random_Forest",
        help="Nombre del experimento en MLflow (default: Paddy_Yield_Random_Forest).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre descriptivo de la corrida en MLflow (opcional).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help=(
            "URI de tracking de MLflow. Si no se indica, se usa MLFLOW_TRACKING_URI, "
            "luego el servidor local en el puerto 5000 y, si no está activo, sqlite:///mlflow.db."
        ),
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Lee el dataset desde un archivo zip (sin extraerlo) o desde un csv."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    if data_path.endswith(".zip"):
        with zipfile.ZipFile(data_path, "r") as z:
            csv_files = [name for name in z.namelist() if name.endswith(".csv")]
            if not csv_files:
                raise ValueError("El archivo zip no contiene ningún csv.")
            csv_name = "paddydataset.csv" if "paddydataset.csv" in csv_files else csv_files[0]
            with z.open(csv_name) as f:
                return pd.read_csv(f)

    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)

    raise ValueError(f"Formato no soportado: {data_path}. Use .zip o .csv.")


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Elimina duplicados y atípicos de la variable objetivo y codifica las categóricas."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"La columna objetivo '{TARGET_COLUMN}' no está en el dataset.")

    df_clean = df.drop_duplicates().reset_index(drop=True)

    q1 = df_clean[TARGET_COLUMN].quantile(0.25)
    q3 = df_clean[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    in_range = df_clean[TARGET_COLUMN].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    df_clean = df_clean[in_range].reset_index(drop=True)

    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(TARGET_COLUMN, axis=1)
    y = df_encoded[TARGET_COLUMN]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento y prueba con una semilla fija."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    random_state: int,
) -> RandomForestRegressor:
    """Entrena el modelo. Se usa un solo hilo para que el resultado sea idéntico entre ejecuciones."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calcula RMSE, MAE, R² y MSE sobre el conjunto de prueba."""
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "mse": mse,
    }
    return y_pred, metrics


def save_and_log_figure(fig: plt.Figure, tmp_dir: str, filename: str) -> None:
    """Guarda una figura en disco, la registra en MLflow bajo plots/ y la cierra."""
    path = os.path.join(tmp_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


def generate_and_log_artifacts(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    run_name: str | None,
) -> None:
    """Genera las gráficas de evaluación y el archivo de métricas y los registra como artefactos."""
    suffix = f" ({run_name})" if run_name else ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Rendimiento real (kg)")
        ax.set_ylabel("Rendimiento predicho (kg)")
        ax.set_title(f"Valores reales vs predichos{suffix}")
        save_and_log_figure(fig, tmp_dir, "actual_vs_predicted.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(y_test - y_pred, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Residual (real - predicho)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de residuales{suffix}")
        save_and_log_figure(fig, tmp_dir, "residuals_distribution.png")

        importances = (
            pd.Series(model.feature_importances_, index=X_test.columns)
            .sort_values()
            .tail(15)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importances.index, importances.values)
        ax.set_xlabel("Importancia relativa")
        ax.set_title(f"Las 15 características más importantes{suffix}")
        save_and_log_figure(fig, tmp_dir, "feature_importance.png")

        metrics_path = os.path.join(tmp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")


def server_is_up(uri: str, timeout: float = 1.0) -> bool:
    """Comprueba si hay un servidor escuchando en el host y puerto de la URI."""
    parsed = urlparse(uri)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def configure_tracking(tracking_uri: str | None) -> None:
    """Selecciona el destino de tracking: argumento, variable de entorno, servidor local o sqlite."""
    if tracking_uri:
        uri = tracking_uri
    elif os.getenv("MLFLOW_TRACKING_URI"):
        uri = os.environ["MLFLOW_TRACKING_URI"]
    elif server_is_up(DEFAULT_SERVER_URI):
        uri = DEFAULT_SERVER_URI
    else:
        uri = LOCAL_TRACKING_URI
    mlflow.set_tracking_uri(uri)


def main() -> None:
    """Ejecuta el flujo completo: datos, entrenamiento, evaluación y registro en MLflow."""
    args = parse_args()
    configure_tracking(args.tracking_uri)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment(args.experiment_name)

    raw_df = load_data(args.data_path)
    X, y = preprocess_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)
    print(f"Registros: {len(raw_df)} originales, {len(X)} tras limpieza; {X.shape[1]} características.")
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "regression",
                "dataset": "paddy_dataset",
                "target": TARGET_COLUMN,
                "algorithm": "RandomForestRegressor",
                "framework": "scikit-learn",
            }
        )

        params = {
            "n_estimators": args.n_estimators,
            "max_depth": "None" if args.max_depth is None else args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "test_size": args.test_size,
        }
        for key, value in params.items():
            mlflow.log_param(key, value)

        model = train_model(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )

        y_pred, metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print("Métricas:", {k: round(v, 4) for k, v in metrics.items()})

        generate_and_log_artifacts(model, X_test, y_test, y_pred, metrics, args.run_name)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.iloc[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"Corrida registrada: {run.info.run_id}")


if __name__ == "__main__":
    main()
