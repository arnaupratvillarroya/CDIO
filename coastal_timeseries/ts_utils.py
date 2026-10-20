"""
ts_utils.py
-----------
Funcions compartides per a la sessio 10 (analisi) i la sessio 12 (prediccio).

Convencio de temps: la serie es mensual (inici de mes) i t = 0, 1, 2, ... (mesos des
de la primera data). Distancia > 0  =  la costa ha retrocedit (erosio);
distancia < 0 = la costa ha avancat (acrecio).
"""
import numpy as np
from scipy.special import expit

TREND_KINDS = ("linear", "step", "sigmoid")


# ------------------------------------------------------------------ metriques
def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = y - yhat
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(r ** 2))),
            "MAE": float(np.mean(np.abs(r))),
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


# ------------------------------------------------------------------ tendencia
def design_matrix(kind, t, t_jump=None, k=None):
    """Columnes: 1, t  [+ esglao (step)]  [+ sigmoide(k (t - t_jump))]."""
    t = np.asarray(t, float)
    cols = [np.ones_like(t), t]
    if kind == "step":
        cols.append((t >= t_jump).astype(float))
    elif kind == "sigmoid":
        cols.append(expit(k * (t - t_jump)))
    elif kind != "linear":
        raise ValueError(kind)
    return np.column_stack(cols)


def fit_trend(kind, t, y, t_jump=None, k=None):
    X = design_matrix(kind, t, t_jump, k)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_trend(spec, t):
    return design_matrix(spec["kind"], t, spec.get("t_jump"), spec.get("k")) @ np.asarray(spec["beta"])


# ------------------------------------------------------------------ cicles (FFT)
def fft_cycles(resid, n_components):
    """Les n_components freqüencies amb mes amplitud del residu (sense la freq. 0).
    Cada component s'escriu com amp * cos(2 pi freq t + phase) i es pot extendre a t > N."""
    resid = np.asarray(resid, float)
    N = len(resid)
    X = np.fft.rfft(resid)
    amp = np.abs(X)
    amp[0] = 0.0
    comps = []
    for k in sorted(np.argsort(amp)[::-1][:n_components]):
        scale = 1.0 / N if (N % 2 == 0 and k == N // 2) else 2.0 / N
        comps.append({"k": int(k), "freq": float(k / N), "period_months": float(N / k),
                      "amp": float(scale * amp[k]), "phase": float(np.angle(X[k]))})
    return comps


def eval_cycles(comps, t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for c in comps:
        out += c["amp"] * np.cos(2 * np.pi * c["freq"] * t + c["phase"])
    return out


# ------------------------------------------------------------------ ajust complet
def n_params(kind, n_cycles):
    base = {"linear": 2, "step": 3, "sigmoid": 3}[kind]
    extra = 2 if kind == "sigmoid" else 0            # t0 i k es busquen a la dades
    return base + extra + 3 * n_cycles               # cada cicle: freq, amplitud, fase


def fit_model(kind, y, n_cycles, t_jump=None, k_grid=None, t0_grid=None):
    """Tendencia (linear / step / sigmoid) + cicles FFT. Retorna un diccionari amb tot
    el necessari per reconstruir i per predir (sessio 12)."""
    y = np.asarray(y, float)
    n = len(y)
    t = np.arange(n, dtype=float)

    def attempt(tj, k):
        beta = fit_trend(kind, t, y, tj, k)
        spec = {"kind": kind, "t_jump": tj, "k": k, "beta": beta.tolist()}
        trend = predict_trend(spec, t)
        comps = fft_cycles(y - trend, n_cycles)
        fitted = trend + eval_cycles(comps, t)
        return float(np.sum((y - fitted) ** 2)), spec, comps

    if kind == "sigmoid":
        t0_grid = t0_grid if t0_grid is not None else [t_jump]
        k_grid = k_grid if k_grid is not None else [0.2, 0.5, 1.0, 2.0, 5.0]
        best = min((attempt(tj, k) for tj in t0_grid for k in k_grid), key=lambda r: r[0])
    else:
        best = attempt(t_jump, None)
    sse, spec, comps = best

    trend = predict_trend(spec, t)
    cyc = eval_cycles(comps, t)
    fitted = trend + cyc
    m = metrics(y, fitted)
    npar = n_params(kind, n_cycles)
    aic = n * np.log(sse / n) + 2 * npar
    bic = n * np.log(sse / n) + npar * np.log(n)
    return {"spec": spec, "cycles": comps, "n_cycles": n_cycles, "metrics": m,
            "AIC": float(aic), "BIC": float(bic), "n_params": npar,
            "trend": trend, "cyclical": cyc, "fitted": fitted, "resid": y - fitted}


def predict_model(model, t_new):
    """Prediccio (sessio 12): tendencia estesa + cicles periodics extesos."""
    t_new = np.asarray(t_new, float)
    return predict_trend(model["spec"], t_new) + eval_cycles(model["cycles"], t_new)