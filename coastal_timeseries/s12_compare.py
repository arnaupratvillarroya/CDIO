#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S12 - Comparació final dels models paramètric Step + FFT i LSTM.

Executar des de coastal_timeseries:
    python s12_compare.py

Entrades esperades a results_s12:
- s12_parametric_predictions.csv
- s12_lstm_predictions.csv

També carrega la sèrie històrica de results_s10 per dibuixar el context.
No torna a entrenar cap model ni modifica results_s10/results_s11.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HISTORICAL_FILE = Path("results_s10/shoreline_timeseries_MS.csv")
PARAMETRIC_FILE = Path("results_s12/s12_parametric_predictions.csv")
LSTM_FILE = Path("results_s12/s12_lstm_predictions.csv")
OUTPUT_DIR = Path("results_s12")


def read_csv_required(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(
            f"No s'ha trobat {path.resolve()}. Executa primer els scripts previs de S12."
        )
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Al fitxer {path} falten columnes {missing}. "
            f"Columnes disponibles: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def calculate_metrics(actual, predicted):
    return {
        "MAE_m": float(mean_absolute_error(actual, predicted)),
        "RMSE_m": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Carregant resultats...")
    historical = read_csv_required(HISTORICAL_FILE, ["date", "distance_m"])
    parametric = read_csv_required(
        PARAMETRIC_FILE,
        ["date", "parametric_prediction", "observed_2025",
         "prediction_lower_95", "prediction_upper_95"],
    )
    lstm = read_csv_required(
        LSTM_FILE, ["date", "lstm_prediction", "observed_2025"]
    )

    # Alineació explícita per data; evita comparar files en ordres diferents.
    comparison = parametric[
        ["date", "parametric_prediction", "observed_2025",
         "prediction_lower_95", "prediction_upper_95"]
    ].merge(
        lstm[["date", "lstm_prediction", "observed_2025"]],
        on="date",
        how="inner",
        suffixes=("_parametric", "_lstm"),
        validate="one_to_one",
    )
    if len(comparison) != 6:
        raise ValueError(
            f"S'esperaven sis mesos comuns de 2025, però se n'han trobat {len(comparison)}."
        )

    actual_param = comparison["observed_2025_parametric"].to_numpy(dtype=float)
    actual_lstm = comparison["observed_2025_lstm"].to_numpy(dtype=float)
    if not np.allclose(actual_param, actual_lstm, rtol=1e-7, atol=1e-7):
        raise ValueError(
            "Els dos fitxers contenen observacions reals 2025 diferents. "
            "Revisa que s'hagin generat amb el mateix conjunt de dades."
        )
    actual = actual_param
    pred_param = comparison["parametric_prediction"].to_numpy(dtype=float)
    pred_lstm = comparison["lstm_prediction"].to_numpy(dtype=float)

    print("\n2) Calculant mètriques sobre els mateixos sis mesos...")
    metrics = {
        "Step + FFT": calculate_metrics(actual, pred_param),
        "LSTM recursiu": calculate_metrics(actual, pred_lstm),
    }
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "model"
    metrics_df.to_csv(OUTPUT_DIR / "s12_model_comparison_metrics.csv")

    print(metrics_df.to_string(float_format=lambda x: f"{x:.3f}"))

    comparison["parametric_error"] = comparison["parametric_prediction"] - actual
    comparison["lstm_error"] = comparison["lstm_prediction"] - actual
    comparison["parametric_absolute_error"] = comparison["parametric_error"].abs()
    comparison["lstm_absolute_error"] = comparison["lstm_error"].abs()
    comparison.to_csv(OUTPUT_DIR / "s12_model_comparison_predictions.csv", index=False)

    # Cobertura empírica dels intervals paramètrics.
    inside = (
        (actual >= comparison["prediction_lower_95"].to_numpy(dtype=float))
        & (actual <= comparison["prediction_upper_95"].to_numpy(dtype=float))
    )
    coverage = {
        "inside_interval": int(inside.sum()),
        "total_observations": int(len(inside)),
        "empirical_coverage_percent": float(100 * inside.mean()),
    }
    with open(OUTPUT_DIR / "s12_model_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "parametric_95_interval_coverage": coverage,
                "validation_period": "2025-01 to 2025-06",
                "n_validation_months": 6,
                "note": (
                    "Both models are evaluated on the same six observed months. "
                    "The LSTM forecast is recursive; the parametric model is extrapolated "
                    "directly. The small validation sample limits generalization claims."
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nCobertura de l'interval paramètric del 95%: "
        f"{coverage['inside_interval']}/{coverage['total_observations']} "
        f"({coverage['empirical_coverage_percent']:.1f}%)"
    )

    print("\n3) Generant gràfica conjunta...")
    historical = historical.dropna(subset=["date", "distance_m"]).sort_values("date")
    hist_dates = historical["date"]
    hist_values = pd.to_numeric(historical["distance_m"], errors="coerce")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(
        hist_dates, hist_values,
        label="Històric observat (2017–2024)", linewidth=1.8,
    )

    dates = comparison["date"]
    lower = comparison["prediction_lower_95"].to_numpy(dtype=float)
    upper = comparison["prediction_upper_95"].to_numpy(dtype=float)

    ax.fill_between(
        dates, lower, upper,
        alpha=0.18, label="Interval de predicció Step + FFT (95%)",
    )
    ax.plot(
        dates, pred_param, "o--",
        label="Pronòstic Step + FFT", linewidth=2,
    )
    ax.plot(
        dates, pred_lstm, "s--",
        label="Pronòstic LSTM recursiu", linewidth=2,
    )
    ax.scatter(
        dates, actual, marker="x", s=75, color="black",
        label="Observacions reals (H1 2025)", zorder=5,
    )

    ax.axvline(pd.Timestamp("2025-01-01"), linestyle=":", color="gray")
    ax.set_title("S12 - Comparació dels pronòstics per al primer semestre de 2025")
    ax.set_xlabel("Data")
    ax.set_ylabel("Distància mitjana de la línia de costa (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s12_final_model_comparison.png", dpi=220)
    plt.close(fig)

    print("\n4) Resultats desats a:")
    print(f"   {OUTPUT_DIR.resolve()}")
    print(" - s12_model_comparison_metrics.csv")
    print(" - s12_model_comparison_predictions.csv")
    print(" - s12_model_comparison_summary.json")
    print(" - s12_final_model_comparison.png")
    print("\nCOMPARACIÓ S12 COMPLETADA")


if __name__ == "__main__":
    main()
