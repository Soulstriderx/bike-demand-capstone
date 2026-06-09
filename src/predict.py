"""
Inference: load the champion model and predict hourly demand.
=============================================================

CONTRACT — implement ``load_model`` and ``predict`` so the Streamlit app can serve
predictions. Keep the signatures.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .train import CHAMPION_PATH, NON_FEATURE_COLUMNS

#: HF Hub repo where the champion model is hosted (avoids pushing 34MB to git).
HF_MODEL_REPO: str = "Soulstrider/bike-demand-model"
HF_MODEL_FILENAME: str = "champion.pkl"


@functools.lru_cache(maxsize=1)
def load_model(path: str = CHAMPION_PATH) -> Any:
    """Load and return the serialised champion model.

    If ``path`` does not exist locally, downloads the model from Hugging Face
    Hub (``HF_MODEL_REPO`` / ``HF_MODEL_FILENAME``) first.

    Args:
        path: Path to the serialised champion (default :data:`CHAMPION_PATH`).

    Returns:
        The loaded model object.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            local_dir=os.path.dirname(path),
        )
        path = downloaded
    return joblib.load(path)


def predict(model: Any, inputs: Dict[str, Any]) -> float:
    """Return predicted demand for a single hour.

    ``inputs`` is a dict of raw user choices from the app (e.g. hour, weekday,
    weathersit, temp, hum, windspeed). This function must transform them into the SAME
    feature shape used at training time, then return a single non-negative number.

    Args:
        model: A loaded champion model.
        inputs: User-supplied feature values.

    Returns:
        Predicted total rentals (``cnt``) for the hour, as a non-negative float.
    """
    row = {
        "season": inputs["season"],
        "yr": inputs["yr"],
        "mnth": inputs["mnth"],
        "hr": inputs["hr"],
        "holiday": inputs["holiday"],
        "weekday": inputs["weekday"],
        "workingday": inputs["workingday"],
        "weathersit": inputs["weathersit"],
        "temp": inputs["temp"],
        "atemp": inputs["atemp"],
        "hum": inputs["hum"],
        "windspeed": inputs["windspeed"],
    }
    df = pd.DataFrame([row])

    df["dayofweek"] = (df["weekday"] + 6) % 7
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)

    X = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns], errors="ignore")

    pred = float(model.predict(X)[0])
    return max(pred, 0.0)
