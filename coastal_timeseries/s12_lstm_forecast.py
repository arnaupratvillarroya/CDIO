#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S12 - Model B: LSTM univariant amb pronòstic recursiu de gener-juny 2025.

Executar des de coastal_timeseries:
    python s12_lstm_forecast.py

El model s'entrena només amb la sèrie històrica 2017-2024. Les observacions
reals de 2025 s'utilitzen exclusivament per avaluar les sis prediccions.
"""
from pathlib import Path
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import ts_utils as U


HISTORICAL_FILE = Path("results_s10/shoreline_timeseries_MS.csv")
OBSERVED_FILE = Path("results_s12/shoreline_observations_2025_MS.csv")
OUTPUT_DIR = Path("results_s12")

WINDOW_SIZE = 12
LSTM_UNITS = 16
MAX_EPOCHS = 200
BATCH_SIZE = 8
PATIENCE = 15
SEED = 42


def load_series(path):
    if not path.exists():
        raise FileNotFoundError(f"No s'ha trobat: {path.resolve()}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "distance_m" not in df.columns:
        raise ValueError(
            f"{path} ha de contenir les columnes date i distance_m. "
            f"Columnes disponibles: {list(df.columns)}"
        )
    dates = pd.to_datetime(df["date"], errors="coerce")
    values = pd.to_numeric(df["distance_m"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    return series


def make_windows(values, window_size):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 1)),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def recursive_forecast(model, history_scaled, n_steps):
    """Prediu un pas i incorpora la predicció a la finestra per al pas següent."""
    window = np.asarray(history_scaled[-WINDOW_SIZE:], dtype=np.float32).copy()
    predictions = []
    for _ in range(n_steps):
        x = window.reshape(1, WINDOW_SIZE, 1)
        next_value = float(model.predict(x, verbose=0)[0, 0])
        predictions.append(next_value)
        window = np.concatenate([window[1:], np.array([next_value], dtype=np.float32)])
    return np.asarray(predictions, dtype=float)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reproductibilitat bàsica dels pesos i de les operacions aleatòries.
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print("1) Carregant dades...")
    historical = load_series(HISTORICAL_FILE)
    observed = load_series(OBSERVED_FILE)
    expected_history = pd.date_range("2017-01-01", periods=96, freq="MS")
    forecast_dates = pd.date_range("2025-01-01", "2025-06-01", freq="MS")

    if len(historical) != 96 or not historical.index.equals(expected_history):
        raise ValueError("La sèrie històrica ha de contenir els 96 mesos de 2017-2024.")
    observed = observed.reindex(forecast_dates)
    if observed.isna().any():
        raise ValueError("Falten observacions reals en algun mes de gener-juny 2025.")

    y = historical.to_numpy(dtype=np.float32)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if std == 0:
        raise ValueError("La desviació estàndard històrica és zero.")
    scaled = (y - mean) / std

    X, target = make_windows(scaled, WINDOW_SIZE)
    # Validació cronològica: últimes 12 finestres reservades per validació.
    split = len(X) - 12
    if split <= 0:
        raise ValueError("No hi ha prou finestres per separar entrenament i validació.")
    X_fit, y_fit = X[:split], target[:split]
    X_val, y_val = X[split:], target[split:]

    print(f"   Observacions històriques: {len(y)}")
    print(f"   Mitjana/STD per normalitzar: {mean:.3f} / {std:.3f} m")
    print(f"   Finestres totals: {len(X)}; entrenament: {len(X_fit)}; validació: {len(X_val)}")
    print(f"   Forma X d'entrenament: {X_fit.shape}; y: {y_fit.shape}")

    print("\n2) Seleccionant el nombre d'èpoques amb validació temporal...")
    keras.backend.clear_session()
    selector = build_model()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history_fit = selector.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1,
    )
    best_epoch = int(np.argmin(history_fit.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history_fit.history["val_loss"]))
    print(f"   Millor època: {best_epoch}; val_loss mínim: {best_val_loss:.4f}")

    print("\n3) Reentrenant amb totes les finestres històriques...")
    # Reentrenem des de zero amb les 84 finestres i el nombre d'èpoques
    # seleccionat en la validació temporal.
    keras.backend.clear_session()
    final_model = build_model()
    final_model.fit(
        X, target,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    print("\n4) Generant el pronòstic recursiu de sis mesos...")
    forecast_scaled = recursive_forecast(final_model, scaled, 6)
    prediction = forecast_scaled * std + mean
    observed_values = observed.to_numpy(dtype=float)
    scores = U.metrics(observed_values, prediction)

    result = pd.DataFrame({
        "date": forecast_dates,
        "lstm_prediction": prediction,
        "observed_2025": observed_values,
    })
    result.to_csv(OUTPUT_DIR / "s12_lstm_predictions.csv", index=False)

    summary = {
        "model": "Univariate LSTM (recursive six-step forecast)",
        "window_size": WINDOW_SIZE,
        "lstm_units": LSTM_UNITS,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_epoch_from_temporal_validation": best_epoch,
        "best_validation_loss_scaled": best_val_loss,
        "training_observations": len(y),
        "training_start": str(historical.index.min().date()),
        "training_end": str(historical.index.max().date()),
        "normalization_mean": mean,
        "normalization_std": std,
        "forecast_strategy": "recursive; each prediction feeds the next input window",
        "metrics_2025": scores,
    }
    with open(OUTPUT_DIR / "s12_lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n5) Avaluació contra les observacions reals...")
    print(f"   MAE:  {scores['MAE']:.3f} m")
    print(f"   RMSE: {scores['RMSE']:.3f} m")
    print(f"   R²:   {scores['R2']:.3f}")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n6) Generant gràfica...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical.index, historical.values, label="Històric 2017–2024")
    ax.plot(forecast_dates, prediction, "o--", label="Pronòstic LSTM recursiu")
    ax.scatter(forecast_dates, observed_values, marker="x", s=55,
               color="black", label="Observacions reals 2025", zorder=5)
    ax.axvline(pd.Timestamp("2025-01-01"), linestyle=":", color="gray")
    ax.set_title("S12 - LSTM recursiu i observacions reals (H1 2025)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Distància mitjana de la línia de costa (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s12_lstm_forecast.png", dpi=200)
    plt.close(fig)

    print(f"\nResultats desats a: {OUTPUT_DIR.resolve()}")
    print(" - s12_lstm_predictions.csv")
    print(" - s12_lstm_metrics.json")
    print(" - s12_lstm_forecast.png")


if __name__ == "__main__":
    main()
