"""
Model training with MLflow experiment tracking.
================================================

CONTRACT — implement ``train_and_log`` so it trains several models, records each as an
MLflow run, selects a champion, and serialises it. Keep the signature.

Run as a module:  ``python -m src.train``
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data_prep import TARGET, add_features, load_and_clean

#: Where the champion model is written.
CHAMPION_PATH: str = "models/champion.pkl"

#: Columns to exclude from features (target, raw datetime).
NON_FEATURE_COLUMNS: list[str] = [TARGET, "dteday"]

#: Models to train.
MODELS: dict[str, object] = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=30, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_and_log(df: pd.DataFrame, champion_path: str = CHAMPION_PATH) -> str:
    """Train 2-3 regressors, log them to MLflow, and save the champion.

    Must:
        - split features/target with NO leakage (``TARGET`` and leakage cols excluded from X),
        - train at least TWO models (e.g. LinearRegression, RandomForestRegressor,
        GradientBoostingRegressor),
        - for EACH model open an ``mlflow.start_run`` and log params + metrics
        (RMSE, MAE, R2) + the model artifact,
        - select the champion by the best held-out RMSE,
        - serialise the champion (e.g. ``joblib.dump``) to ``champion_path``.

    Args:
        df: Feature-engineered DataFrame (post :func:`add_features`).
        champion_path: Destination path for the serialised champion.

    Returns:
        The path the champion was written to.
    """
    X = df.drop(columns=NON_FEATURE_COLUMNS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    Path(champion_path).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("bike_demand")

    best_rmse = float("inf")
    best_model = None

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = _metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  {name:20s}  RMSE={metrics['rmse']:.2f}  "
                f"MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained.")

    joblib.dump(best_model, champion_path)
    print(f"\nChampion saved to {champion_path}  (RMSE={best_rmse:.2f})")
    return champion_path


def main(data_path: Optional[str] = None) -> str:
    """Entry point: clean -> feature-engineer -> train_and_log."""
    path = data_path or "data/hour.csv"
    df = add_features(load_and_clean(path))
    return train_and_log(df)


if __name__ == "__main__":
    print(f"Champion saved to: {main()}")
