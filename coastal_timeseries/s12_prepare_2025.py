#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S12 - Preparació de les observacions reals de gener a juny de 2025.

Aplica a 2025 els criteris definits a S10 sense introduir informació futura
en la selecció dels transectes ni en els llindars d'outliers:
  1) Selecció de la finestra consecutiva de 100 transectes amb menys NaN
     utilitzant exclusivament les dades històriques 2017-2024.
  2) Conservació dels transectes amb <= 30% de NaN en l'històric.
  3) Llindars IQR (k=1.5) calculats per transecte amb l'històric i aplicats
     també a 2025.
  4) Eliminació de dates amb > 40% de NaN entre els transectes seleccionats.
  5) Mitjana transversal i agregació mensual (MS), com a S10.

Executar des de la carpeta coastal_timeseries:
    python s12_prepare_2025.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def detect_column(columns, candidates, description):
    """Troba una columna acceptant diferents noms habituals."""
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"No s'ha pogut identificar la columna de {description}. "
        f"Columnes disponibles: {list(columns)}"
    )


def load_pivot(csv_path):
    """Carrega el CSV i crea una matriu data x transecte."""
    df = pd.read_csv(csv_path)

    date_col = detect_column(
        df.columns, ["date", "datetime", "time", "timestamp"], "data"
    )
    transect_col = detect_column(
        df.columns, ["transect_id", "transect", "id_transect"], "transecte"
    )
    distance_col = detect_column(
        df.columns,
        ["distance_m", "distance", "shoreline_distance", "distancia_m"],
        "distància",
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[transect_col] = pd.to_numeric(df[transect_col], errors="coerce")
    df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
    df = df.dropna(subset=[date_col, transect_col])

    pivot = df.pivot_table(
        index=date_col,
        columns=transect_col,
        values=distance_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot = pivot.dropna(axis=0, how="all")
    return pivot


def select_historical_transects(
    historical, n_transects=100, max_nan_transect=0.30
):
    """Replica la selecció S10 només amb dades històriques."""
    nan_fraction = historical.isna().mean(axis=0)

    # Cerca la finestra consecutiva de n transectes amb menys NaN mitjà.
    ids = np.asarray(historical.columns, dtype=float)
    if len(ids) < n_transects:
        raise ValueError(
            f"L'històric només conté {len(ids)} transectes; "
            f"no es poden seleccionar {n_transects}."
        )

    # Una finestra és consecutiva quan els identificadors són consecutius.
    best_window = None
    best_score = np.inf
    for start in range(len(ids) - n_transects + 1):
        window_ids = ids[start : start + n_transects]
        if not np.allclose(np.diff(window_ids), 1):
            continue
        score = float(nan_fraction.loc[window_ids].mean())
        if score < best_score:
            best_score = score
            best_window = window_ids

    if best_window is None:
        raise ValueError(
            "No s'ha trobat cap finestra de transectes amb identificadors "
            "consecutius. Cal revisar els IDs o adaptar la selecció."
        )

    selected = [
        tid
        for tid in best_window
        if nan_fraction.loc[tid] <= max_nan_transect
    ]
    if not selected:
        raise ValueError("Cap transecte compleix el llindar històric de NaN.")
    return selected, best_window.tolist(), best_score


def calculate_iqr_limits(historical_selected, iqr_k=1.5):
    """Calcula límits IQR per transecte a partir de 2017-2024."""
    q1 = historical_selected.quantile(0.25)
    q3 = historical_selected.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    return lower, upper


def main():
    parser = argparse.ArgumentParser(
        description="Prepara les observacions mensuals reals de 2025 per a S12."
    )
    parser.add_argument(
        "--historical",
        default="../data/input/shoreline_distances_2017_2024.csv",
        help="CSV històric utilitzat a S10.",
    )
    parser.add_argument(
        "--observed-2025",
        default="../data/input/shoreline_distances_h1_2025.csv",
        help="CSV amb les observacions de gener a juny de 2025.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_s12",
        help="Carpeta on es desaran els resultats.",
    )
    parser.add_argument("--n-transects", type=int, default=100)
    parser.add_argument("--max-nan-transect", type=float, default=0.30)
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--max-nan-date", type=float, default=0.40)
    args = parser.parse_args()

    historical_path = Path(args.historical)
    observed_path = Path(args.observed_2025)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (historical_path, observed_path):
        if not path.exists():
            raise FileNotFoundError(
                f"No s'ha trobat {path.resolve()}. "
                "Comprova que executes el script des de coastal_timeseries "
                "o indica la ruta amb els arguments."
            )

    print("1) Carregant dades històriques i dades de 2025...")
    hist = load_pivot(historical_path)
    obs_2025 = load_pivot(observed_path)
    print(
        f"   Històric: {hist.shape[0]} dates x {hist.shape[1]} transectes "
        f"({hist.index.min().date()} a {hist.index.max().date()})"
    )
    print(
        f"   2025: {obs_2025.shape[0]} dates x {obs_2025.shape[1]} transectes "
        f"({obs_2025.index.min().date()} a {obs_2025.index.max().date()})"
    )

    print("\n2) Seleccionant els transectes exclusivament amb l'històric...")
    selected, window, window_nan = select_historical_transects(
        hist, args.n_transects, args.max_nan_transect
    )
    selected_columns = pd.Index(selected)
    hist_selected = hist.reindex(columns=selected_columns)
    obs_selected = obs_2025.reindex(columns=selected_columns)

    missing_columns = [tid for tid in selected if tid not in obs_2025.columns]
    if missing_columns:
        raise ValueError(
            "El CSV de 2025 no conté tots els transectes seleccionats a S10. "
            f"Falten: {missing_columns}"
        )

    print(f"   Finestra històrica: {window[0]}–{window[-1]}")
    print(f"   NaN mitjà de la finestra: {window_nan:.1%}")
    print(f"   Transectes conservats: {len(selected)}")

    print("\n3) Calculant llindars IQR amb 2017-2024 i aplicant-los a 2025...")
    lower, upper = calculate_iqr_limits(hist_selected, args.iqr_k)
    cleaned_2025 = obs_selected.copy()
    for tid in selected:
        outside = (cleaned_2025[tid] < lower.loc[tid]) | (
            cleaned_2025[tid] > upper.loc[tid]
        )
        cleaned_2025.loc[outside, tid] = np.nan

    outliers_removed = int(obs_selected.notna().sum().sum() -
                           cleaned_2025.notna().sum().sum())
    print(f"   Valors eliminats com a outliers: {outliers_removed}")

    print("\n4) Aplicant el llindar de cobertura per data...")
    nan_by_date = cleaned_2025.isna().mean(axis=1)
    coverage_after_iqr = nan_by_date.copy()
    keep_dates = nan_by_date <= args.max_nan_date
    cleaned_2025 = cleaned_2025.loc[keep_dates]
    removed_dates = int((~keep_dates).sum())
    print(f"   Dates eliminades (> {args.max_nan_date:.0%} NaN): {removed_dates}")

    print("\n5) Calculant la mitjana diària i la sèrie mensual...")
    daily_mean = cleaned_2025.mean(axis=1)
    monthly = daily_mean.resample("MS").mean()

    # Manté només els mesos amb observacions; no crea mesos fora del període.
    monthly = monthly.loc[monthly.first_valid_index():monthly.last_valid_index()]
    monthly = monthly.interpolate(method="linear", limit_area="inside")
    monthly.index.name = "date"
    monthly.name = "distance_m"

    expected_months = pd.date_range("2025-01-01", "2025-06-01", freq="MS")
    monthly = monthly.reindex(expected_months)
    monthly.index.name = "date"
    monthly.name = "distance_m"

    result = monthly.reset_index()
    result.to_csv(output_dir / "shoreline_observations_2025_MS.csv", index=False)

    selected_df = pd.DataFrame({"transect_id": selected})
    selected_df.to_csv(output_dir / "selected_transects_used.csv", index=False)

    limits = pd.DataFrame({
        "transect_id": selected,
        "iqr_lower": [lower.loc[tid] for tid in selected],
        "iqr_upper": [upper.loc[tid] for tid in selected],
    })
    limits.to_csv(output_dir / "historical_iqr_limits.csv", index=False)

    coverage = pd.DataFrame({
        "date": obs_selected.index,
        "nan_fraction_before_iqr": obs_selected.isna().mean(axis=1).values,
    })
    coverage["nan_fraction_after_iqr"] = (
        coverage_after_iqr.reindex(obs_selected.index).values
    )
    coverage["date_kept"] = coverage["date"].isin(cleaned_2025.index)
    coverage.to_csv(output_dir / "daily_coverage_2025.csv", index=False)

    summary = {
        "historical_file": str(historical_path),
        "observed_2025_file": str(observed_path),
        "historical_window_first_transect": float(window[0]),
        "historical_window_last_transect": float(window[-1]),
        "historical_window_mean_nan_fraction": float(window_nan),
        "n_selected_transects": len(selected),
        "selected_transects": [float(x) for x in selected],
        "iqr_k": args.iqr_k,
        "max_nan_date": args.max_nan_date,
        "outliers_removed_2025": outliers_removed,
        "dates_removed_for_low_coverage": removed_dates,
        "monthly_observations": {
            str(row.date.date()): (
                None if pd.isna(row.distance_m) else float(row.distance_m)
            )
            for row in result.itertuples(index=False)
        },
    }
    with open(output_dir / "s12_prepare_2025_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nRESULTAT MENSUAL 2025")
    print(result.to_string(index=False))
    print(f"\nResultats desats a: {output_dir.resolve()}")
    print("Fitxer principal: shoreline_observations_2025_MS.csv")
    if result["distance_m"].isna().any():
        print(
            "\nATENCIÓ: hi ha mesos sense observació després del filtratge. "
            "No els interpretis com a valors reals; revisarem la cobertura "
            "abans de fer la validació."
        )


if __name__ == "__main__":
    main()
