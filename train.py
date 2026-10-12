"""
Entrenamiento parametrizado e instrumentado con MLflow Tracking.

Predice el costo anual del seguro médico (`charges`, USD) del dataset
"Medical Cost Personal" con un Random Forest (o la regresión lineal baseline
de la actividad anterior) y registra cada corrida en MLflow.

Uso rápido (valores por defecto):
    python train.py

Ejemplo variando hiperparámetros:
    python train.py --n-estimators 300 --max-depth 5 --min-samples-leaf 5 --seed 42

Ver todos los argumentos:
    python train.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "charges"
NUM_COLS = ["age", "bmi", "children"]
CAT_COLS = ["sex", "smoker", "region"]
COLOR_NO, COLOR_SI, COLOR_REF = "#2a78d6", "#eb6834", "#52514e"


# --------------------------------------------------------------------------- #
# Argumentos de línea de comandos
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee los argumentos. Todos tienen valor por defecto."""
    p = argparse.ArgumentParser(
        description="Entrena un modelo de costos de seguro médico y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Modelo e hiperparámetros
    p.add_argument("--model", choices=["random_forest", "linear"], default="random_forest",
                   help="Algoritmo: random_forest o linear (baseline de la actividad anterior).")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Número de árboles del bosque (solo random_forest).")
    p.add_argument("--max-depth", type=int, default=4,
                   help="Profundidad máxima de cada árbol; 0 = sin límite (solo random_forest).")
    p.add_argument("--min-samples-leaf", type=int, default=1,
                   help="Mínimo de registros por hoja; valores altos regularizan (solo random_forest).")
    p.add_argument("--max-features", type=float, default=1.0,
                   help="Fracción de variables evaluadas en cada división, entre 0 y 1 (solo random_forest).")
    # Datos, evaluación y reproducibilidad
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Particiones de validación cruzada sobre entrenamiento; 0 la desactiva.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para división, CV y modelo.")
    p.add_argument("--data-path", default="data/insurance.csv", help="Ruta al CSV del dataset.")
    # MLflow
    p.add_argument("--experiment-name", default="TC5061_Costos_Seguro_Medico",
                   help="Experimento de MLflow donde se agrupan las corridas.")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                   help="Almacén de MLflow (archivo SQLite local o URL de un servidor, p. ej. http://127.0.0.1:5000).")
    p.add_argument("--run-name", default=None, help="Nombre de la corrida; si se omite se genera uno descriptivo.")
    args = p.parse_args(argv)

    # Validaciones tempranas para fallar con un mensaje claro
    if not 0 < args.test_size < 1:
        p.error("--test-size debe estar entre 0 y 1")
    if not 0 < args.max_features <= 1:
        p.error("--max-features debe estar en (0, 1]")
    if args.n_estimators < 1 or args.min_samples_leaf < 1 or args.max_depth < 0:
        p.error("--n-estimators y --min-samples-leaf deben ser >= 1 y --max-depth >= 0")
    return args


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad de Python y NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def file_md5(path: str) -> str:
    """Huella del archivo de datos para saber con qué versión se entrenó."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    """Lee el CSV y comprueba que tenga las columnas esperadas."""
    df = pd.read_csv(path)
    faltan = set(NUM_COLS + CAT_COLS + [TARGET]) - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {path}: {sorted(faltan)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las decisiones de limpieza documentadas en el notebook de exploración."""
    df = df.drop_duplicates().reset_index(drop=True)          # 1 duplicado exacto
    for col in CAT_COLS:                                     # texto normalizado y tipo categoría
        df[col] = df[col].astype(str).str.strip().str.lower().astype("category")
    # Los outliers de charges (fumadores) y de bmi se conservan: son casos reales
    if df[NUM_COLS + [TARGET]].isna().any().any():
        raise ValueError("Hay valores faltantes en columnas numéricas")
    return df


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """División entrenamiento / prueba reproducible."""
    X, y = df[NUM_COLS + CAT_COLS], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Preprocesamiento + estimador en un solo Pipeline (evita fuga de información)."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ])
    if args.model == "linear":
        estimator = LinearRegression()
    else:
        estimator = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or None,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.seed,
            n_jobs=-1,
        )
    return Pipeline([("prep", prep), ("model", estimator)])


def model_params(args: argparse.Namespace) -> dict:
    """Hiperparámetros efectivos que se registran en MLflow."""
    params = {"model": args.model, "test_size": args.test_size, "cv_folds": args.cv_folds, "seed": args.seed}
    if args.model == "random_forest":
        params.update({"n_estimators": args.n_estimators,
                       "max_depth": args.max_depth or "None",
                       "min_samples_leaf": args.min_samples_leaf,
                       "max_features": args.max_features})
    return params


def default_run_name(args: argparse.Namespace) -> str:
    if args.model == "linear":
        return "linear_baseline"
    depth = args.max_depth or "none"
    return f"rf_n{args.n_estimators}_d{depth}_leaf{args.min_samples_leaf}_mf{args.max_features:g}"


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred, prefix: str) -> dict:
    return {f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_r2": float(r2_score(y_true, y_pred))}


def cross_validate(pipeline: Pipeline, X, y, folds: int, seed: int) -> dict:
    """RMSE con validación cruzada en entrenamiento: estima qué tan estable es el resultado."""
    if folds < 2:
        return {}
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return {"cv_rmse_mean": float(scores.mean()), "cv_rmse_std": float(scores.std())}


# --------------------------------------------------------------------------- #
# Artefactos
# --------------------------------------------------------------------------- #
def plot_predictions(y_true, y_pred, smoker, path: Path) -> None:
    """Real vs predicho y residuos, coloreados por fumador."""
    colors = np.where(np.asarray(smoker) == "yes", COLOR_SI, COLOR_NO)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    axes[0].scatter(y_true, y_pred, c=colors, s=20, alpha=.8, edgecolors="white", linewidths=.4)
    axes[0].plot(lim, lim, ls="--", lw=1.3, color=COLOR_REF)
    axes[0].set(xlim=lim, ylim=lim, title="Real vs predicho (test)", xlabel="charges real (USD)",
                ylabel="charges predicho (USD)")
    axes[1].scatter(y_pred, np.asarray(y_true) - y_pred, c=colors, s=20, alpha=.8, edgecolors="white", linewidths=.4)
    axes[1].axhline(0, ls="--", lw=1.3, color=COLOR_REF)
    axes[1].set(title="Residuos (test)", xlabel="charges predicho (USD)", ylabel="residuo (USD)")
    for ax in axes:
        ax.grid(color="#e6e5e1"); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in [(COLOR_NO, "no fuma"), (COLOR_SI, "fuma")]]
    axes[0].legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(pipeline: Pipeline, path: Path) -> None:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    est = pipeline.named_steps["model"]
    values = getattr(est, "feature_importances_", None)
    label = "Importancia (reducción de impureza)"
    if values is None:
        values, label = np.abs(est.coef_), "|coeficiente| (USD)"
    serie = pd.Series(values, index=[n.split("__", 1)[1] for n in names]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(serie.index, serie.values, color=COLOR_NO, height=.6)
    ax.set(title="Importancia de variables", xlabel=label)
    ax.grid(axis="x", color="#e6e5e1"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)

    df = clean_data(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)
    pipeline = build_pipeline(args)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or default_run_name(args)) as run:
        # 1) Parámetros de la corrida
        for key, value in model_params(args).items():
            mlflow.log_param(key, value)
        mlflow.log_param("data_md5", file_md5(args.data_path))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # 2) Entrenamiento y métricas
        cv_metrics = cross_validate(pipeline, X_train, y_train, args.cv_folds, args.seed)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = {**regression_metrics(y_test, y_pred, "test"),
                   **regression_metrics(y_train, pipeline.predict(X_train), "train"),
                   **cv_metrics}
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # 3) Artefactos: gráficas + archivo de métricas
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_predictions(y_test.to_numpy(), y_pred, X_test["smoker"].astype(str), tmp / "real_vs_predicho.png")
            plot_importance(pipeline, tmp / "importancia_variables.png")
            (tmp / "metrics.json").write_text(json.dumps({"params": model_params(args), "metrics": metrics}, indent=2))
            for name in ["real_vs_predicho.png", "importancia_variables.png", "metrics.json"]:
                mlflow.log_artifact(str(tmp / name), artifact_path="evaluacion")

        # 4) Modelo entrenado con firma de entrada/salida.
        mlflow.sklearn.log_model(pipeline, name="model",
                                 signature=infer_signature(X_train, pipeline.predict(X_train)),
                                 input_example=X_train.head(3),
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"Run: {run.info.run_name} ({run.info.run_id})")
        for key, value in metrics.items():
            print(f"  {key:<14} {value:,.4f}")
    return metrics


if __name__ == "__main__":
    main()
