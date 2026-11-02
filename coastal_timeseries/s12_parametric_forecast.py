#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S12 Model A: pronòstic paramètric Step + FFT per a gener-juny de 2025.

Executar des de coastal_timeseries:
    python s12_parametric_forecast.py

El model s'ajusta només amb 2017-2024. Les dades reals de 2025 s'usen
exclusivament per validar el pronòstic.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ts_utils as U

HISTORICAL_FILE = Path("results_s10/shoreline_timeseries_MS.csv")
OBSERVED_FILE = Path("results_s12/shoreline_observations_2025_MS.csv")
OUTPUT_DIR = Path("results_s12")
N_CYCLES = 1
T_JUMP = 37
FORECAST_DATES = pd.date_range("2025-01-01", "2025-06-01", freq="MS")


def load_series(path, value_candidates):
    if not path.exists():
        raise FileNotFoundError(f"No s'ha trobat: {path.resolve()}")
    df = pd.read_csv(path)
    date_col = next((c for c in ("date", "datetime", "time")
                     if c in df.columns), None)
    value_col = next((c for c in value_candidates if c in df.columns), None)
    if date_col is None or value_col is None:
        raise ValueError(f"Columnes no identificades a {path}: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=value_col).dropna().sort_index()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("1) Carregant les sèries...")
    historical = load_series(
        HISTORICAL_FILE, ("distance_m", "mean_distance_m", "shoreline_distance")
    )
    observed = load_series(
        OBSERVED_FILE, ("distance_m", "mean_distance_m", "shoreline_distance")
    )

    expected_history = pd.date_range("2017-01-01", periods=96, freq="MS")
    if len(historical) != 96 or not historical.index.equals(expected_history):
        raise ValueError("La sèrie històrica ha de contenir els 96 mesos de 2017-2024.")
    observed = observed.reindex(FORECAST_DATES)
    if observed.isna().any():
        raise ValueError("Falten observacions reals en algun mes de gener-juny 2025.")

    print(f"   Històric: {len(historical)} mesos; observacions 2025: {len(observed)} mesos")
    print("\n2) Ajustant Step + FFT amb les 96 observacions històriques...")
    model = U.fit_model("step", historical.to_numpy(float), N_CYCLES, t_jump=T_JUMP)
    t_future = np.arange(96, 102, dtype=float)
    prediction = U.predict_model(model, t_future)

    # Mateixa aproximació d'interval de predicció emprada a S11.
    sigma = float(np.std(model["resid"], ddof=1))
    h = np.arange(1, 7, dtype=float)
    margin = 1.96 * sigma * np.sqrt(1.0 + h / len(historical))
    lower, upper = prediction - margin, prediction + margin

    print(f"   RMSE d'ajust històric: {model['metrics']['RMSE']:.3f} m")
    print(f"   Desviació residual: {sigma:.3f} m")
    print("\n3) Validant contra les observacions reals de 2025...")
    observed_values = observed.to_numpy(float)
    scores = U.metrics(observed_values, prediction)
    inside = (observed_values >= lower) & (observed_values <= upper)

    result = pd.DataFrame({
        "date": FORECAST_DATES,
        "parametric_prediction": prediction,
        "prediction_lower_95": lower,
        "prediction_upper_95": upper,
        "observed_2025": observed_values,
        "inside_95_interval": inside,
    })
    result.to_csv(OUTPUT_DIR / "s12_parametric_predictions.csv", index=False)

    summary = {
        "model": "Step + FFT",
        "n_cycles": N_CYCLES,
        "t_jump": T_JUMP,
        "training_observations": len(historical),
        "residual_std": sigma,
        "prediction_interval_formula": "yhat +/- 1.96 * sigma * sqrt(1 + h/n)",
        "metrics_2025": scores,
        "observations_inside_95_interval": int(inside.sum()),
        "total_observations": len(inside),
        "empirical_interval_coverage": float(inside.mean()),
    }
    with open(OUTPUT_DIR / "s12_parametric_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   MAE:  {scores['MAE']:.3f} m")
    print(f"   RMSE: {scores['RMSE']:.3f} m")
    print(f"   R²:   {scores['R2']:.3f}")
    print(f"   Observacions dins l'interval: {inside.sum()}/6 ({inside.mean():.1%})")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n4) Generant la gràfica...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical.index, historical.values, label="Històric 2017–2024")
    ax.plot(FORECAST_DATES, prediction, "o--", label="Pronòstic Step + FFT")
    ax.fill_between(FORECAST_DATES, lower, upper, alpha=0.2,
                    label="Interval de predicció 95%")
    ax.scatter(FORECAST_DATES, observed_values, marker="x", s=55,
               color="black", label="Observacions reals 2025", zorder=5)
    ax.axvline(pd.Timestamp("2025-01-01"), linestyle=":", color="gray")
    ax.set_title("S12 - Pronòstic paramètric i observacions reals (H1 2025)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Distància mitjana de la línia de costa (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s12_parametric_forecast.png", dpi=200)
    plt.close(fig)

    print(f"\nResultats desats a: {OUTPUT_DIR.resolve()}")
    print(" - s12_parametric_predictions.csv")
    print(" - s12_parametric_metrics.json")
    print(" - s12_parametric_forecast.png")


if __name__ == "__main__":
    main()
