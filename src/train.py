import argparse
import json
import os
import tempfile
import zipfile
from typing import Dict, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento y seguimiento de modelo RandomForestRegressor para rendimiento de arroz (Paddy Yield)."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Numero de arboles en el ensamble (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=lambda v: None if v in ("None", "none", "null", "") else int(v),
        default=None,
        help="Profundidad maxima de los arboles (default: None).",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Numero minimo de muestras requeridas para dividir un nodo (default: 2).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Numero minimo de muestras en un nodo hoja (default: 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad del split y modelo (default: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporcion del conjunto de prueba (default: 0.2).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/paddy+dataset.zip",
        help="Ruta al archivo de datos en formato zip o csv (default: data/paddy+dataset.zip).",
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
        help="Nombre descriptivo para la ejecucion en MLflow (opcional).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="URI del servidor de MLflow (default: variable MLFLOW_TRACKING_URI o servidor local).",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontro el archivo de datos en la ruta: {data_path}")

    if data_path.endswith(".zip"):
        with zipfile.ZipFile(data_path, "r") as z:
            csv_candidates = [name for name in z.namelist() if name.endswith(".csv")]
            if not csv_candidates:
                raise ValueError("No se encontro ningun archivo CSV dentro del archivo zip.")
            csv_filename = "paddydataset.csv" if "paddydataset.csv" in csv_candidates else csv_candidates[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {data_path}. Use .zip o .csv.")

    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_column = "Paddy yield(in Kg)"
    if target_column not in df.columns:
        raise KeyError(f"La columna objetivo '{target_column}' no esta presente en el DataFrame.")

    # Eliminacion de registros duplicados
    df_clean = df.drop_duplicates().reset_index(drop=True)

    # Tratamiento de valores atipicos en la variable objetivo mediante Rango Intercuartilico (IQR)
    q1 = df_clean[target_column].quantile(0.25)
    q3 = df_clean[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_clean = df_clean[(df_clean[target_column] >= lower_bound) & (df_clean[target_column] <= upper_bound)]
    df_clean = df_clean.reset_index(drop=True)

    # Codificacion de variables categoricas mediante One-Hot Encoding
    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": float(mse),
    }
    return y_pred, metrics


def generate_and_log_artifacts(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    run_name: str | None,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Grafica de Real vs Predicho
        fig_pred, ax_pred = plt.subplots(figsize=(7, 6))
        ax_pred.scatter(y_test, y_pred, alpha=0.5, color="#1f77b4", edgecolors="none")
        min_val = min(float(y_test.min()), float(y_pred.min()))
        max_val = max(float(y_test.max()), float(y_pred.max()))
        ax_pred.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=1.5)
        ax_pred.set_xlabel("Rendimiento Real (kg)")
        ax_pred.set_ylabel("Rendimiento Predicho (kg)")
        title_suffix = f" ({run_name})" if run_name else ""
        ax_pred.set_title(f"Valores Reales vs Predichos{title_suffix}")
        fig_pred.tight_layout()
        pred_path = os.path.join(tmp_dir, "actual_vs_predicted.png")
        fig_pred.savefig(pred_path, dpi=150)
        plt.close(fig_pred)
        mlflow.log_artifact(pred_path, artifact_path="plots")

        # 2. Distribucion de residuales
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(7, 5))
        ax_res.hist(residuals, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax_res.axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
        ax_res.set_xlabel("Residual (Real - Predicho)")
        ax_res.set_ylabel("Frecuencia")
        ax_res.set_title(f"Distribucion de Residuales{title_suffix}")
        fig_res.tight_layout()
        res_path = os.path.join(tmp_dir, "residuals_distribution.png")
        fig_res.savefig(res_path, dpi=150)
        plt.close(fig_res)
        mlflow.log_artifact(res_path, artifact_path="plots")

        # 3. Importancia de caracteristicas (Top 15)
        feature_importances = model.feature_importances_
        features = list(X_test.columns)
        feat_df = (
            pd.DataFrame({"caracteristica": features, "importancia": feature_importances})
            .sort_values("importancia", ascending=True)
            .tail(15)
        )

        fig_fi, ax_fi = plt.subplots(figsize=(9, 6))
        ax_fi.barh(feat_df["caracteristica"], feat_df["importancia"], color="#1f77b4")
        ax_fi.set_xlabel("Importancia Relativa")
        ax_fi.set_title(f"Top 15 Caracteristicas mas Influyentes{title_suffix}")
        fig_fi.tight_layout()
        fi_path = os.path.join(tmp_dir, "feature_importance.png")
        fig_fi.savefig(fi_path, dpi=150)
        plt.close(fig_fi)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # 4. Archivo de metricas en formato JSON
        metrics_path = os.path.join(tmp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")


def configure_tracking(tracking_uri: str | None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        default_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(default_uri)


def main():
    args = parse_args()
    configure_tracking(args.tracking_uri)

    print(f"Tracking URI configurada: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment(args.experiment_name)

    # 1. Carga y preprocesamiento de datos
    print(f"Cargando dataset desde: {args.data_path}")
    raw_df = load_data(args.data_path)
    print(f"Dataset original cargado con {raw_df.shape[0]} filas y {raw_df.shape[1]} columnas.")

    X, y = preprocess_data(raw_df)
    print(f"Dataset procesado: {X.shape[0]} filas y {X.shape[1]} caracteristicas resultantes tras one-hot encoding.")

    # 2. Division de conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Muestras de entrenamiento: {len(X_train)} | Muestras de evaluacion: {len(X_test)}")

    # 3. Registro de experimento en MLflow
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        effective_run_name = args.run_name if args.run_name else run_id[:8]
        print(f"MLflow Run ID: {run_id} ({effective_run_name})")

        # Tags descriptivos
        mlflow.set_tags(
            {
                "task": "regression",
                "dataset": "paddy_dataset",
                "target": "Paddy yield(in Kg)",
                "model_family": "ensemble",
                "algorithm": "RandomForestRegressor",
                "framework": "scikit-learn",
            }
        )

        # Registro de hiperparametros
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": "None" if args.max_depth is None else args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "test_size": args.test_size,
        }
        mlflow.log_params(params)
        print("Parametros registrados:", params)

        # 4. Entrenamiento
        print("Entrenando modelo RandomForestRegressor...")
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )

        # 5. Evaluacion
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        print("Metricas de desempeno obtenidas:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 6. Artefactos (graficas y reporte numerico)
        generate_and_log_artifacts(
            model=model,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics,
            run_name=args.run_name,
        )

        # 7. Registro del modelo empaquetado con firma y ejemplo de entrada
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print("Modelo, parametros, metricas y artefactos registrados exitosamente en MLflow.")
        print(f"Artefactos ubicados en: {run.info.artifact_uri}")


if __name__ == "__main__":
    main()
