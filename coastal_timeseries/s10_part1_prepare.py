"""
s10_part1_prepare.py  --  Sessio 10, Part 1: preparacio de dades
----------------------------------------------------------------
Del CSV brut (data, transect_id, distance_m) a una serie temporal neta i regular.

    python s10_part1_prepare.py --input ruta/shoreline_distances_2017_2024.csv

Pasos (numeracio de l'enunciat):
  1-4  carregar, inspeccionar, convertir dates, pivot (data x transecte)
  5    % de NaN i distribucio per transecte (i per data / mes)
  6-7  subconjunt de ~100 transectes consecutius amb pocs NaN + filtre per llindar
  8    outliers per transecte (IQR)
  9    dates amb massa poca cobertura
  10   serie agregada (mitjana dels transectes)
  11   comparacio de remostreig (setmanal, quinzenal, mensual)
  12-13 interpolacio lineal i desat a CSV
Tot el que es genera va a --outdir (CSV, JSON i figures).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def nan_gaps(s):
    """Llista de (inici, fi, longitud) dels trams consecutius de NaN."""
    isn = s.isna().to_numpy()
    gaps, i = [], 0
    while i < len(isn):
        if isn[i]:
            j = i
            while j + 1 < len(isn) and isn[j + 1]:
                j += 1
            gaps.append((s.index[i], s.index[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return gaps


def best_window(nan_frac, w):
    """w transectes consecutius (= tram de costa contigu) amb el menor % de NaN mitja."""
    w = min(w, len(nan_frac))
    roll = nan_frac.rolling(w).mean().dropna()
    end = roll.idxmin()
    pos = nan_frac.index.get_loc(end)
    return nan_frac.index[pos - w + 1: pos + 1]


def main():
    ap = argparse.ArgumentParser(description="Sessio 10 - Part 1: preparacio de dades")
    ap.add_argument("--input", type=Path,
                    default=HERE.parent / "data" / "input" / "shoreline_distances.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "results_s10")
    ap.add_argument("--n-transects", type=int, default=100, help="mida del tram de costa (nombre de transectes)")
    ap.add_argument("--max-nan-transect", type=float, default=0.30, help="llindar de NaN per transecte")
    ap.add_argument("--iqr-k", type=float, default=1.5, help="factor IQR per als outliers")
    ap.add_argument("--max-nan-date", type=float, default=0.40, help="llindar de NaN per data")
    ap.add_argument("--freq", default="MS", help="freq. final: MS (mensual), 2W, W")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No trobo {args.input}. Passeu la ruta amb --input.")
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------------------------------------------------------- 1-4. carregar i pivotar
    print("=" * 70 + "\n1-4. CARREGAR I INSPECCIONAR\n" + "=" * 70)
    df = pd.read_csv(args.input)
    c_date = find_col(df, ["date", "datetime", "time"])
    c_tr = find_col(df, ["transect_id", "transect", "id"])
    c_dist = find_col(df, ["distance_m", "distance", "dist_m", "dist"])
    if None in (c_date, c_tr, c_dist):
        raise SystemExit(f"No trobo les columnes data/transecte/distancia. Columnes: {list(df.columns)}")
    print(df.head(), "\n", df.tail(), sep="")
    df.info()
    df[c_date] = pd.to_datetime(df[c_date])
    print(f"\nRang de dates: {df[c_date].min().date()} -> {df[c_date].max().date()}")

    piv = df.pivot_table(index=c_date, columns=c_tr, values=c_dist, aggfunc="mean").sort_index()
    piv = piv.reindex(sorted(piv.columns), axis=1)
    n_dates_all = len(piv)
    piv = piv.dropna(how="all")                   # una data sense cap mesura no es una observacio
    print(f"Pivot: {piv.shape[0]} dates x {piv.shape[1]} transectes "
          f"({n_dates_all - len(piv)} dates sense cap valor eliminades)")
    summary.update(n_rows_raw=len(df), n_dates=int(piv.shape[0]), n_transects=int(piv.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[c_date], df[c_dist], ",", alpha=0.3)
    ax.set(title="Dades brutes (tots els transectes)", xlabel="Data", ylabel="Distancia (m)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "01_raw_overview.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 5. NaN
    print("\n" + "=" * 70 + "\n5. VALORS PERDUTS\n" + "=" * 70)
    nan_t = piv.isna().mean()
    nan_d = piv.isna().mean(axis=1)
    print(f"% NaN global: {piv.isna().values.mean():.1%} | per transecte: "
          f"mediana {nan_t.median():.1%}, min {nan_t.min():.1%}, max {nan_t.max():.1%}")
    by_month = nan_d.groupby(nan_d.index.month).agg(["mean", "count"])
    by_month.columns = ["nan_mitja", "n_dates"]
    print("\nNaN per mes natural (esta concentrat en una epoca?):\n", by_month.round(2).to_string())
    by_year = nan_d.groupby(nan_d.index.year).agg(["mean", "count"])
    by_year.columns = ["nan_mitja", "n_dates"]
    print("\nNaN per any:\n", by_year.round(2).to_string())

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].hist(nan_t * 100, bins=30)
    axs[0].set(title="Distribucio del % de NaN per transecte", xlabel="% NaN", ylabel="# transectes")
    axs[1].plot(nan_t.index, nan_t * 100, lw=0.8)
    axs[1].set(title="% de NaN segons la posicio (transect_id)", xlabel="transect_id", ylabel="% NaN")
    axs[2].plot(nan_d.index, nan_d * 100, ".", ms=3)
    axs[2].set(title="% de NaN per data", xlabel="Data", ylabel="% NaN")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "02_missing.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 6-7. seleccio de transectes
    print("\n" + "=" * 70 + "\n6-7. SELECCIO DE TRANSECTES\n" + "=" * 70)
    window = best_window(nan_t, args.n_transects)
    sub = piv[window]
    nan_sub = sub.isna().mean()
    keep = nan_sub[nan_sub <= args.max_nan_transect].index
    print(f"Tram de {len(window)} transectes consecutius: ids {window[0]} -> {window[-1]} "
          f"(NaN mitja {nan_sub.mean():.1%})")
    print(f"Despres del llindar de {args.max_nan_transect:.0%}: {len(keep)} transectes "
          f"({len(window) - len(keep)} eliminats)")
    if len(keep) == 0:
        raise SystemExit("Cap transecte supera el llindar de NaN: pugeu --max-nan-transect "
                         f"(el tram escollit te {nan_sub.mean():.0%} de NaN mitja).")
    if len(keep) < 20:
        print(f"[AVIS] nomes {len(keep)} transectes: la serie sera soroll. Considereu pujar --max-nan-transect.")
    sub = sub[keep]
    summary.update(window_ids=[int(window[0]), int(window[-1])], n_transects_kept=int(len(keep)))

    # ---------------------------------------------------------- 8. outliers
    print("\n" + "=" * 70 + "\n8. OUTLIERS (IQR per transecte)\n" + "=" * 70)
    q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - args.iqr_k * iqr, q3 + args.iqr_k * iqr
    is_out = sub.lt(lo, axis=1) | sub.gt(hi, axis=1)
    n_out = int(is_out.values.sum())
    print(f"Outliers eliminats: {n_out} ({n_out / sub.notna().values.sum():.2%} dels valors)")
    summary["n_outliers_removed"] = n_out
    raw_sub = sub.copy()
    sub = sub.mask(is_out)

    ex = raw_sub.notna().sum().sort_values(ascending=False).index[:3]
    fig, axs = plt.subplots(len(ex), 1, figsize=(12, 2.6 * len(ex)), sharex=True)
    for a, tid in zip(np.atleast_1d(axs), ex):
        a.plot(raw_sub.index, raw_sub[tid], "o", ms=3, color="tab:red", label="outlier")
        a.plot(sub.index, sub[tid], "o", ms=3, color="tab:blue", label="vàlid")
        a.set_ylabel(f"transecte {tid}")
        a.grid(alpha=0.3)
    np.atleast_1d(axs)[0].legend(ncol=2)
    np.atleast_1d(axs)[0].set_title("Exemples de filtre d'outliers (vermell = eliminat)")
    fig.tight_layout(); fig.savefig(out / "03_outliers.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------- 9. dates amb poca cobertura
    print("\n" + "=" * 70 + "\n9. DATES AMB POCA COBERTURA\n" + "=" * 70)
    nan_date = sub.isna().mean(axis=1)
    good = nan_date <= args.max_nan_date
    print(f"Dates amb mes de {args.max_nan_date:.0%} de NaN eliminades: {int((~good).sum())} de {len(sub)}")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(nan_date.index, (1 - nan_date) * 100, ".", ms=4)
    ax.axhline((1 - args.max_nan_date) * 100, color="r", ls="--", label="llindar de cobertura")
    ax.set(title="Cobertura per data (% de transectes amb valor)", ylabel="% cobertura")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "04_date_coverage.png", dpi=130); plt.close(fig)
    sub = sub[good]
    summary["n_dates_kept"] = int(len(sub))

    # ---------------------------------------------------------- 10. serie agregada
    agg = sub.mean(axis=1)
    agg.name = "distance_m"
    print(f"\n10. Serie agregada (mitjana de {sub.shape[1]} transectes): {len(agg)} dates")

    # ---------------------------------------------------------- 11. remostreig
    print("\n" + "=" * 70 + "\n11. REMOSTREIG\n" + "=" * 70)
    rows = []
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, f in zip(axs, ["W", "2W", "MS"]):
        r = agg.resample(f).mean()
        r = r.loc[r.first_valid_index():r.last_valid_index()]
        gaps = nan_gaps(r)
        rows.append({"freq": f, "n_punts": len(r), "pct_NaN": r.isna().mean() * 100,
                     "tram_NaN_mes_llarg": max([g[2] for g in gaps], default=0), "std_m": r.std()})
        ax.plot(r.index, r.values, ".-", lw=0.8, ms=3)
        ax.set_title(f"freq = {f}  ({r.isna().mean():.0%} NaN)")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "05_resampling.png", dpi=130); plt.close(fig)
    comp = pd.DataFrame(rows).set_index("freq")
    print(comp.round(2).to_string())
    comp.round(3).to_csv(out / "resampling_comparison.csv")

    # ---------------------------------------------------------- 12-13. interpolar i desar
    print("\n" + "=" * 70 + f"\n12-13. INTERPOLACIO I DESAT (freq = {args.freq})\n" + "=" * 70)
    r = agg.resample(args.freq).mean()
    r = r.loc[r.first_valid_index():r.last_valid_index()]        # no extrapolem les puntes
    was_nan = r.isna()
    gaps = nan_gaps(r)
    long_gaps = [g for g in gaps if g[2] > 2]
    print(f"Punts a interpolar: {int(was_nan.sum())} de {len(r)} ({was_nan.mean():.1%})")
    print(f"Trams de NaN: {len(gaps)} | trams llargs (> 2 punts): {len(long_gaps)}")
    for s, e, n in long_gaps:
        print(f"   {s.date()} -> {e.date()}  ({n} punts)  <-- l'interpolacio pot amagar un forat real")
    ts = r.interpolate(method="linear", limit_area="inside")
    final = pd.DataFrame({"date": ts.index, "distance_m": ts.values, "interpolated": was_nan.astype(int).values})
    fname = f"shoreline_timeseries_{args.freq}.csv"
    final.to_csv(out / fname, index=False)
    pd.Series(sub.columns, name="transect_id").to_csv(out / "selected_transects.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts.index, ts.values, "-", color="tab:blue", lw=1, label="serie final")
    ax.plot(ts.index[~was_nan.values], ts.values[~was_nan.values], "o", ms=4, color="tab:blue", label="observat")
    ax.plot(ts.index[was_nan.values], ts.values[was_nan.values], "o", ms=5, color="tab:red", label="interpolat")
    ax.set(title="Serie temporal final (mitjana dels transectes seleccionats)", ylabel="Distancia (m)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "06_final_series.png", dpi=130); plt.close(fig)

    summary.update(freq=args.freq, n_points=int(len(ts)), n_interpolated=int(was_nan.sum()),
                   long_gaps=[[str(s.date()), str(e.date()), int(n)] for s, e, n in long_gaps])
    (out / "part1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSerie final desada a {out / fname}")
    print("Resum:", json.dumps(summary))


if __name__ == "__main__":
    main()