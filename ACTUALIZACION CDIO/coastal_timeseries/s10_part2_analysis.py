"""
s10_part2_analysis.py  --  Sessio 10, Part 2: analisi de la serie temporal
--------------------------------------------------------------------------
    python s10_part2_analysis.py                      # fa servir results_s10/shoreline_timeseries_MS.csv
    python s10_part2_analysis.py --input altra.csv --jump 2020-02-01 --n-cycles 3

Pasos:  2 grafic | 3 estadistics | 4 boxplots | 5 tendencia lineal | 6 FFT (periodes)
        7-9 original vs reconstruida, metriques, residus (3 models: linear / step / sigmoid)
        10-11 velocitat i acceleracio del model escollit.
Desa el model escollit a best_model.json (el reutilitza la sessio 12).

Convencio: distancia > 0 = la costa retrocedeix (EROSIO); < 0 = avanca (ACRECIO).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ts_utils as U

HERE = Path(__file__).resolve().parent
COLORS = {"linear": "tab:green", "step": "tab:orange", "sigmoid": "tab:purple"}


def sign_runs(idx, x, label_pos, label_neg, min_len=2, tol=0.0, label_zero="~0"):
    """Trams consecutius amb el mateix signe: (inici, fi, etiqueta, valor mitja).
    Els valors amb |x| <= tol es consideren zero."""
    x = np.asarray(x)
    s = np.where(x > tol, 1, np.where(x < -tol, -1, 0))
    labels = {1: label_pos, -1: label_neg, 0: label_zero}
    runs, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if i - start >= min_len:
                runs.append((idx[start], idx[i - 1], labels[int(s[start])], float(np.mean(x[start:i]))))
            start = i
    return runs


def robust_ylim(ax, x, pct=(2, 98), pad=0.8):
    lo, hi = np.percentile(x, pct)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - pad * span, hi + pad * span)


def acf(x, nlags):
    x = np.asarray(x) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[: len(x) - k] * x[k:]) / d for k in range(nlags + 1)])


def main():
    ap = argparse.ArgumentParser(description="Sessio 10 - Part 2: analisi")
    ap.add_argument("--input", type=Path, default=HERE / "results_s10" / "shoreline_timeseries_MS.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "results_s10")
    ap.add_argument("--jump", default="2020-02-01", help="data del canvi estructural (tempesta Gloria)")
    ap.add_argument("--n-cycles", type=int, default=3, help="components FFT que es mantenen")
    ap.add_argument("--model", choices=list(U.TREND_KINDS), default=None,
                    help="forca el model escollit (per defecte: el de menor BIC)")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1-2. carregar i dibuixar
    ts = pd.read_csv(args.input, parse_dates=["date"], index_col="date")["distance_m"]
    if pd.infer_freq(ts.index) != "MS":
        print("[AVIS] la serie no es mensual (MS): es remostreja a mensual i s'interpola.")
        ts = ts.resample("MS").mean().interpolate()
    y = ts.to_numpy(float)
    n = len(y)
    t = np.arange(n, dtype=float)
    print(f"Serie: {n} mesos, {ts.index[0].date()} -> {ts.index[-1].date()}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts.index, y, "o-", ms=3, lw=1)
    ax.set(title="Serie temporal de la distancia de la costa", ylabel="Distancia (m)  [+ = erosio]")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "10_series.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ 3. estadistics
    desc = ts.describe()
    desc["skew"], desc["kurtosis"] = stats.skew(y), stats.kurtosis(y)
    print("\n3. ESTADISTICS DESCRIPTIUS\n", desc.round(3).to_string())
    desc.round(4).to_csv(out / "descriptive_stats.csv", header=["value"])

    # ------------------------------------------------------------ 5. tendencia lineal
    lr = stats.linregress(t, y)
    ci = 1.96 * lr.stderr * 12
    print(f"\n5. TENDENCIA LINEAL: {lr.slope * 12:+.2f} m/any (IC95% +-{ci:.2f}) | R2={lr.rvalue ** 2:.3f} | "
          f"p={lr.pvalue:.2g}  ->  " + ("la costa RETROCEDEIX (erosio)" if lr.slope > 0 else "la costa AVANCA (acrecio)"))
    lin_detr = y - (lr.intercept + lr.slope * t)

    # ------------------------------------------------------------ 4. boxplots
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    months = [y[ts.index.month == m] for m in range(1, 13)]
    axs[0].boxplot(months, tick_labels=range(1, 13)); axs[0].set(title="Per mes (serie original)", xlabel="Mes")
    years = sorted(set(ts.index.year))
    axs[1].boxplot([y[ts.index.year == a] for a in years], tick_labels=years)
    axs[1].set(title="Per any (serie original)", xlabel="Any")
    axs[2].boxplot([lin_detr[ts.index.month == m] for m in range(1, 13)], tick_labels=range(1, 13))
    axs[2].set(title="Per mes (sense tendencia lineal)", xlabel="Mes")
    for a in axs:
        a.grid(alpha=0.3); a.set_ylabel("m")
    fig.tight_layout(); fig.savefig(out / "11_boxplots.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ 6-9. models
    t_jump = int(ts.index.get_indexer([pd.Timestamp(args.jump)], method="nearest")[0])
    t0_grid = list(range(max(0, t_jump - 3), min(n - 1, t_jump + 3) + 1))
    k_grid = [0.2, 0.5, 1.0, 2.0, 5.0]

    print("\n6. BIC segons el nombre de components FFT (menys = millor):")
    sweep = {kind: [U.fit_model(kind, y, c, t_jump, k_grid, t0_grid)["BIC"] for c in range(1, 9)]
             for kind in U.TREND_KINDS}
    sweep_df = pd.DataFrame(sweep, index=range(1, 9)).rename_axis("n_cycles")
    print(sweep_df.round(1).to_string())
    sweep_df.round(2).to_csv(out / "bic_vs_ncycles.csv")

    models = {k: U.fit_model(k, y, args.n_cycles, t_jump, k_grid, t0_grid) for k in U.TREND_KINDS}
    rows = []
    for k, m in models.items():
        sp = m["spec"]
        if k == "step":
            extra = f"salt = {sp['beta'][2]:+.2f} m a {ts.index[t_jump].date()}"
        elif k == "sigmoid":
            extra = (f"amplitud = {sp['beta'][2]:+.2f} m, t0 = {ts.index[int(sp['t_jump'])].date()}, "
                     f"k = {sp['k']}")
        else:
            extra = ""
        rows.append({"model": k, **{a: round(b, 3) for a, b in m["metrics"].items()},
                     "AIC": round(m["AIC"], 1), "BIC": round(m["BIC"], 1), "n_params": m["n_params"],
                     "detall_tendencia": extra})
    comp = pd.DataFrame(rows).set_index("model")
    print(f"\n7-8. COMPARACIO DE MODELS (tendencia + {args.n_cycles} cicles FFT)\n", comp.to_string())
    comp.to_csv(out / "model_comparison.csv")

    best_kind = args.model or comp["BIC"].idxmin()
    best = models[best_kind]
    print(f"\n==> Model escollit: {best_kind} (menor BIC: {comp['BIC'].idxmin()}). "
          f"Cal justificar-lo tambe fisicament (Gloria = canvi brusc o gradual?).")

    for k, m in models.items():
        pd.DataFrame(m["cycles"]).drop(columns=["k"]).to_csv(out / f"fft_components_{k}.csv", index=False)
    print(f"\n6. COMPONENTS FFT del model {best_kind}:")
    print(pd.DataFrame(best["cycles"])[["period_months", "amp"]].assign(
        period_years=lambda d: d.period_months / 12).round(2).sort_values("amp", ascending=False).to_string(index=False))

    # espectre
    resid_tr = y - best["trend"]
    X = np.abs(np.fft.rfft(resid_tr)) * 2 / n
    period = np.r_[np.inf, n / np.arange(1, len(X))]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(period[1:], X[1:], basefmt=" ")
    for c in best["cycles"]:
        ax.plot(c["period_months"], c["amp"], "ro")
    ax.axvline(12, color="gray", ls="--", label="12 mesos")
    ax.set(xscale="log", xlabel="Periode (mesos)", ylabel="Amplitud (m)",
           title=f"Espectre FFT del residu de la tendencia ({best_kind}); vermell = components mantinguts")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "12_fft_spectrum.png", dpi=130); plt.close(fig)

    # original vs reconstruida
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, (k, m) in zip(axs, models.items()):
        ax.plot(ts.index, y, color="tab:blue", alpha=0.45, lw=1, label="original")
        ax.plot(ts.index, m["trend"], color="tab:orange", label="tendencia")
        ax.plot(ts.index, m["cyclical"], color="tab:red", lw=1, label="cicles")
        ax.plot(ts.index, m["fitted"], color="purple", lw=2.2, label="reconstruida")
        ax.set_title(f"{k}:  RMSE={m['metrics']['RMSE']:.2f} m  MAE={m['metrics']['MAE']:.2f} m  "
                     f"R2={m['metrics']['R2']:.3f}  BIC={m['BIC']:.1f}")
        ax.set_ylabel("m"); ax.grid(alpha=0.3); ax.legend(ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "13_models.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ 9. residus
    r = best["resid"]
    sh = stats.shapiro(r)
    dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
    print(f"\n9. RESIDUS ({best_kind}): mitjana={r.mean():+.3f} m, std={r.std():.2f} m, "
          f"Shapiro p={sh.pvalue:.3f} ({'compatible amb normal' if sh.pvalue > 0.05 else 'NO normal'}), "
          f"Durbin-Watson={dw:.2f} (~2 = sense autocorrelacio)")
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
    axs[0].hist(r, bins=20, edgecolor="k"); axs[0].axvline(0, color="r", ls="--")
    axs[0].set(title="Distribucio dels residus", xlabel="m")
    axs[1].plot(ts.index, r, ".-"); axs[1].axhline(0, color="r", ls="--"); axs[1].set(title="Residus en el temps")
    a_ = acf(r, 24)
    axs[2].bar(range(25), a_); axs[2].axhline(1.96 / np.sqrt(n), color="r", ls="--")
    axs[2].axhline(-1.96 / np.sqrt(n), color="r", ls="--"); axs[2].set(title="Autocorrelacio dels residus", xlabel="lag (mesos)")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "14_residuals.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ 10-11. velocitat i acceleracio
    dt = 1.0                                        # 1 mes entre mostres
    rec = best["fitted"]
    vel = np.gradient(rec, dt) * 12                 # m/any
    acc = np.gradient(np.gradient(rec, dt), dt) * 144  # m/any^2
    pd.DataFrame({"date": ts.index, "reconstructed_m": rec, "velocity_m_year": vel,
                  "acceleration_m_year2": acc}).to_csv(out / "erosion_dynamics.csv", index=False)
    vel_tr = np.gradient(best["trend"], dt) * 12               # nomes la tendencia (sense estacionalitat)
    acc_tr = np.gradient(np.gradient(best["trend"], dt), dt) * 144
    pd.DataFrame({"date": ts.index, "reconstructed_m": rec, "velocity_m_year": vel,
                  "acceleration_m_year2": acc, "trend_velocity_m_year": vel_tr,
                  "trend_acceleration_m_year2": acc_tr}).to_csv(out / "erosion_dynamics.csv", index=False)
    print(f"\n10-11. DINAMICA (model {best_kind}): velocitat mitjana {vel.mean():+.2f} m/any | "
          f"pic |v| = {np.abs(vel).max():.1f} m/any a {ts.index[int(np.abs(vel).argmax())].date()}")
    runs_v = sign_runs(ts.index, vel, "erosio", "acrecio")
    print(f"\nSerie reconstruida: {len(runs_v)} trams alternants erosio/acrecio (en general, el cicle estacional "
          f"domina la velocitat instantania).")
    print("Velocitat de la TENDENCIA (sense cicles), tram a tram:")
    for s_, e_, lab, val in sign_runs(ts.index, vel_tr, "erosio", "acrecio", min_len=1, tol=0.05):
        print(f"   {s_.date()} -> {e_.date()}  {lab:8s}  v mitjana {val:+.2f} m/any")
    print("Acceleracio de la TENDENCIA (a>0: la velocitat augmenta; a<0: disminueix):")
    for s_, e_, lab, val in sign_runs(ts.index, acc_tr, "a>0", "a<0", min_len=1, tol=0.05):
        print(f"   {s_.date()} -> {e_.date()}  {lab}  a mitjana {val:+.2f} m/any^2")
    print("(Si v>0, a>0 = l'EROSIO s'accelera; a<0 = s'alenteix. Si v<0, el signe de l'acrecio s'inverteix.)")

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(ts.index, rec, color="purple"); axs[0].set_ylabel("Distancia reconstruida (m)")
    axs[1].plot(ts.index, vel, color="tab:blue", alpha=0.6, label="reconstruida")
    axs[1].plot(ts.index, vel_tr, color="k", lw=2, label="nomes tendencia"); axs[1].axhline(0, color="k", lw=0.8)
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("Velocitat (m/any)\n>0 erosio, <0 acrecio")
    axs[2].plot(ts.index, acc, color="tab:red", alpha=0.6); axs[2].plot(ts.index, acc_tr, color="k", lw=2)
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel("Acceleracio (m/any$^2$)")
    for a, x, name in ((axs[1], vel, "v"), (axs[2], acc, "a")):
        robust_ylim(a, x)
        if np.abs(x).max() > max(abs(a.get_ylim()[0]), abs(a.get_ylim()[1])):
            a.text(0.01, 0.92, f"pic de {name} retallat a la grafica: {x[np.abs(x).argmax()]:+.0f}",
                   transform=a.transAxes, color="gray", fontsize=9)
    for a in axs:
        a.axvline(pd.Timestamp(args.jump), color="gray", ls=":", label="Gloria" if a is axs[0] else None)
        a.grid(alpha=0.3)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(out / "15_velocity_acceleration.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ desar model escollit
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    spec = clean(best["spec"])
    spec["t_jump"] = None if spec["t_jump"] is None else int(spec["t_jump"])
    payload = {"kind": best_kind, "spec": spec, "cycles": clean(best["cycles"]), "n_cycles": args.n_cycles,
               "metrics": clean(best["metrics"]), "start_date": str(ts.index[0].date()), "freq": "MS",
               "n_train": n, "residual_std": float(np.std(best["resid"], ddof=1))}
    (out / "best_model.json").write_text(json.dumps(payload, indent=2))
    print(f"\nModel escollit desat a {out / 'best_model.json'} (es reutilitza a la sessio 12)")


if __name__ == "__main__":
    main()