"""
experiments_morphology.py
-------------------------
Barrido de  (denoising de entrada)  x  (operadores morfologicos sobre la mascara
de agua)  y calculo del RMSE contra el ground truth en las 3 fechas de validacion.

Colocar junto a cdio_pipeline.py (carpeta "ACTUALIZACION CDIO") y ejecutar:

    python experiments_morphology.py --quick      # primera prueba (rapida)
    python experiments_morphology.py              # rejilla completa

Necesita que ya esten descargadas las bandas en data/images_mt_1/
(<escena>_green.tif y <escena>_nir.tif; las genera ndwi_n_photose.py).

Salidas (data/output/experiments/):
    experiments_results.csv   una fila por (fecha, denoising, morfologia)
    experiments_summary.csv   ranking por configuracion
    rmse_heatmap.png          RMSE medio (denoising x morfologia)
    qualitative_<fecha>.png   GT vs linea estimada (baseline / mejores)
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from scipy.interpolate import splev, splprep
from scipy.ndimage import (binary_closing, binary_erosion, binary_opening,
                           convolve, label, uniform_filter)
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral

# ==========================================
# CONFIGURACIO
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
BANDS_DIR = BASE_DIR / "data" / "images_mt_1"
OUT_DIR = BASE_DIR / "data" / "output" / "experiments"
GT_DIRS = [BASE_DIR / "ground_truth", BASE_DIR / "data" / "input" / "ground_truth"]

GT_DATES = ["2017-05-25", "2019-05-23", "2021-06-09"]

TRANSECT_LENGTH = 300      # m (+-150 m a cada costat)
EDGE_MARGIN = 5            # px que descartem al voltant de les vores / nodata
REFLECTANCE_SCALE = 10000.0
RMSE_TARGET = 50.0


# ==========================================
# 1. DENOISING (entrada)  -- bandes en reflectancia [0, 1]
# ==========================================
def bilateral(a, sigma_color):
    return denoise_bilateral(a, sigma_color=sigma_color, sigma_spatial=2, channel_axis=None)


DENOISERS = {
    "none": lambda a: a,
    "boxcar_3x3": lambda a: uniform_filter(a, size=3, mode="reflect"),
    "boxcar_4x4": lambda a: uniform_filter(a, size=4, mode="reflect"),
    "boxcar_5x5": lambda a: uniform_filter(a, size=5, mode="reflect"),
    "bilateral_sc0.02": lambda a: bilateral(a, 0.02),
    "bilateral_sc0.05": lambda a: bilateral(a, 0.05),
    "bilateral_sc0.10": lambda a: bilateral(a, 0.10),
}


# ==========================================
# 2. OPERADORS MORFOLOGICS (sortida: mascara d'aigua)
# ==========================================
def remove_small(mask, min_area):
    """Elimina components connexos (True) de menys de min_area pixels."""
    lab, n = label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_area
    keep[0] = False
    return keep[lab]


def area_filter(mask, area):
    """Treu illes d'aigua petites i omple forats de terra petits."""
    mask = remove_small(mask, area)
    return ~remove_small(~mask, area)


def open_(mask, r):
    return binary_opening(mask, structure=disk(r))


def close_(mask, r):
    return binary_closing(mask, structure=disk(r))


MORPHS = {
    "none": lambda m: m,
    "open_r1": lambda m: open_(m, 1),
    "open_r2": lambda m: open_(m, 2),
    "close_r1": lambda m: close_(m, 1),
    "close_r2": lambda m: close_(m, 2),
    "open_close_r1": lambda m: close_(open_(m, 1), 1),
    "open_close_r2": lambda m: close_(open_(m, 2), 2),
    "area_50": lambda m: area_filter(m, 50),
    "area_200": lambda m: area_filter(m, 200),
}

QUICK_DENOISERS = ["none", "boxcar_4x4", "bilateral_sc0.05"]
QUICK_MORPHS = ["none", "open_r1", "close_r1", "open_close_r1", "area_50"]


# ==========================================
# 3. ESCENES I GROUND TRUTH
# ==========================================
def find_gt_file(date_str):
    key = date_str.replace("-", "")
    for d in GT_DIRS:
        if d.is_dir():
            hits = sorted(p for p in d.glob("*.csv") if key in p.name)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No trobo cap CSV de ground truth amb la data {date_str} a {GT_DIRS}")


def scene_date(scene_id):
    m = re.search(r"(\d{8})", scene_id)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def closest_scene(date_str):
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    cands = []
    for g in BANDS_DIR.glob("*_green.tif"):
        sid = g.name[: -len("_green.tif")]
        d = scene_date(sid)
        if d and (BANDS_DIR / f"{sid}_nir.tif").exists():
            cands.append((abs((d - target).days), d, sid))
    if not cands:
        raise FileNotFoundError(f"No hi ha bandes green/nir a {BANDS_DIR}")
    gap, d, sid = min(cands)
    return sid, d, gap


def candidate_scenes(date_str, max_gap):
    """Totes les escenes descarregades a <= max_gap dies de la data (mes properes primer)."""
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    out = []
    for g in BANDS_DIR.glob("*_green.tif"):
        sid = g.name[: -len("_green.tif")]
        d = scene_date(sid)
        if d and (BANDS_DIR / f"{sid}_nir.tif").exists() and abs((d - target).days) <= max_gap:
            out.append((sid, d, abs((d - target).days)))
    return sorted(out, key=lambda t: (t[2], t[0]))


def load_scene(scene_id):
    with rasterio.open(BANDS_DIR / f"{scene_id}_green.tif") as g, \
            rasterio.open(BANDS_DIR / f"{scene_id}_nir.tif") as n:
        green = g.read(1).astype(np.float32)
        nir = n.read(1).astype(np.float32)
        transform = g.transform
    valid = (green > 0) | (nir > 0)                      # fora de l'AOI = 0
    inner = binary_erosion(valid, structure=np.ones((3, 3), bool),
                           iterations=EDGE_MARGIN, border_value=0)
    green = np.clip(green / REFLECTANCE_SCALE, 0, 1)
    nir = np.clip(nir / REFLECTANCE_SCALE, 0, 1)
    return {"green": green, "nir": nir, "valid": valid, "inner": inner, "transform": transform}


def build_gt(gt_path, spacing):
    """Linia de referencia, transectes cada `spacing` m i normals (mantenint l'ordre)."""
    df = pd.read_csv(gt_path, sep=None, engine="python")
    xy = df.select_dtypes(include=[np.number]).iloc[:, :2].dropna().to_numpy(float)
    _, idx = np.unique(xy, axis=0, return_index=True)
    xy = xy[np.sort(idx)]                                # sense reordenar per x
    length = np.hypot(*np.diff(xy, axis=0).T).sum()
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0)
    u = np.linspace(0, 1, max(2, int(length / spacing)))
    px, py = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    normals = np.column_stack((-dy, dx))
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    normals /= norms[:, None]
    pts = np.column_stack((px, py))
    return {"xy": xy, "pts": pts, "normals": normals, "tree": cKDTree(pts), "length": length}


def footprint_stats(gt, scene):
    """Quin % de la costa GT cau dins la imatge i sobre pixels valids (no nodata)."""
    cols, rows = (~scene["transform"]) * (gt["pts"][:, 0], gt["pts"][:, 1])
    h, w = scene["valid"].shape
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    ok = np.zeros(len(cols), bool)
    ok[inside] = scene["inner"][rows[inside].astype(int), cols[inside].astype(int)]
    return float(inside.mean()), float(ok.mean())


# ==========================================
# 4. LINIA DE COSTA I DISTANCIES
# ==========================================
NEIGH_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def water_mask(green, nir, valid):
    denom = green + nir
    ndwi = np.where(denom > 0, (green - nir) / np.where(denom > 0, denom, 1.0), 0.0)
    return (ndwi > 0) & valid


def shoreline_contours(mask, inner, transform, selection):
    """Mateix criteri que coastline_estimator.py + generate_shorelines_csv.py
    (pixels d'aigua amb algun veí de terra -> find_contours)."""
    neighbor_sum = convolve(mask.astype(np.uint8), NEIGH_KERNEL, mode="constant", cval=0)
    coast = mask & (neighbor_sum < 8) & inner
    if not coast.any():
        return []
    contours = [c for c in find_contours(coast.astype(np.float32), 0.5) if len(c) >= 3]
    if selection == "longest" and contours:
        contours = [max(contours, key=len)]
    out = []
    for c in contours:
        xs, ys = transform * (c[:, 1] + 0.5, c[:, 0] + 0.5)
        out.append(np.column_stack((xs, ys)))
    return out


def signed_distances(gt, contours, half=TRANSECT_LENGTH / 2):
    """Distancia (amb signe, sobre la normal) de cada transecte a la interseccio
    mes propera amb la costa estimada. NaN si no n'hi ha."""
    n_t = len(gt["pts"])
    out = np.full(n_t, np.nan)
    if not contours:
        return out
    a = np.concatenate([c[:-1] for c in contours])
    b = np.concatenate([c[1:] for c in contours])
    d, _ = gt["tree"].query((a + b) / 2, distance_upper_bound=half + 20)
    keep = np.isfinite(d)
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return out
    e = b - a
    chunk = 128
    for s0 in range(0, n_t, chunk):
        p = gt["pts"][s0:s0 + chunk][:, None, :]
        n = gt["normals"][s0:s0 + chunk][:, None, :]
        ap = a[None, :, :] - p
        denom = n[..., 0] * e[None, :, 1] - n[..., 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (ap[..., 0] * e[None, :, 1] - ap[..., 1] * e[None, :, 0]) / denom
            t = (ap[..., 0] * n[..., 1] - ap[..., 1] * n[..., 0]) / denom
        ok = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (np.abs(s) <= half)
        abs_s = np.where(ok, np.abs(s), np.inf)
        j = abs_s.argmin(axis=1)
        rows = np.arange(len(j))
        found = np.isfinite(abs_s[rows, j])
        out[s0:s0 + chunk] = np.where(found, s[rows, j], np.nan)
    return out


def metrics(dist):
    v = dist[np.isfinite(dist)]
    if len(v) == 0:
        return {"rmse": np.nan, "bias": np.nan, "mae": np.nan, "n_valid": 0, "coverage": 0.0}
    return {"rmse": float(np.sqrt(np.mean(v ** 2))), "bias": float(v.mean()),
            "mae": float(np.abs(v).mean()), "n_valid": int(len(v)),
            "coverage": float(len(v) / len(dist))}


def process(scene, dn_name, mp_name, selection, cache=None):
    """denoising -> NDWI > 0 -> morfologia -> contorns."""
    if cache is not None and dn_name in cache:
        water0 = cache[dn_name]
    else:
        dn = DENOISERS[dn_name]
        water0 = water_mask(dn(scene["green"]), dn(scene["nir"]), scene["valid"])
        if cache is not None:
            cache[dn_name] = water0
    mask = MORPHS[mp_name](water0)
    return mask, shoreline_contours(mask, scene["inner"], scene["transform"], selection)


# ==========================================
# 5. FIGURES
# ==========================================
def plot_heatmap(res, dn_order, mp_order, path):
    piv = res.groupby(["denoising", "morphology"]).rmse.mean().unstack()
    piv = piv.reindex(index=dn_order, columns=mp_order)
    fig, ax = plt.subplots(figsize=(1.2 * len(mp_order) + 2, 0.7 * len(dn_order) + 2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(mp_order)), mp_order, rotation=40, ha="right")
    ax.set_yticks(range(len(dn_order)), dn_order)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("RMSE mitja (m) de les 3 dates: denoising x morfologia")
    fig.colorbar(im, ax=ax, label="m")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(date_str, gt, scene, configs, res, selection, path):
    cmap = ListedColormap(["#e8dcb5", "#9ec9e8"])
    h, w = scene["valid"].shape
    left, top = scene["transform"] * (0, 0)
    right, bottom = scene["transform"] * (w, h)
    cx, cy = gt["pts"][len(gt["pts"]) // 2]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (label_txt, (dn, mp)) in zip(axs.ravel(), configs):
        mask, contours = process(scene, dn, mp, selection)
        ax.imshow(mask, cmap=cmap, extent=(left, right, bottom, top), interpolation="nearest")
        ax.plot(gt["xy"][:, 0], gt["xy"][:, 1], "r-", lw=1.5, label="Ground truth")
        for k, c in enumerate(contours):
            ax.plot(c[:, 0], c[:, 1], "k-", lw=0.8, label="Estimada" if k == 0 else None)
        ax.set_xlim(cx - 350, cx + 350)
        ax.set_ylim(cy - 220, cy + 220)
        ax.set_aspect("equal")
        row = res[(res.gt_date == date_str) & (res.denoising == dn) & (res.morphology == mp)]
        r = row.rmse.iloc[0] if len(row) else float("nan")
        ax.set_title(f"{label_txt}: {dn} + {mp}\nRMSE = {r:.1f} m", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"GT {date_str}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Denoising x morfologia -> RMSE vs ground truth")
    ap.add_argument("--gt-dates", nargs="+", default=GT_DATES)
    ap.add_argument("--spacing", type=float, default=10.0, help="separacio entre transectes (m)")
    ap.add_argument("--selection", choices=["all", "longest"], default="all",
                    help="all = tots els contorns; longest = nomes el mes llarg (com generate_shorelines_csv.py)")
    ap.add_argument("--max-gap", type=int, default=20,
                    help="maxim de dies entre GT i imatge S2; si es superior, es salta la data")
    ap.add_argument("--quick", action="store_true", help="rejilla reduida")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    dn_order = QUICK_DENOISERS if args.quick else list(DENOISERS)
    mp_order = QUICK_MORPHS if args.quick else list(MORPHS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, gts, scenes, info, skipped = [], {}, {}, {}, {}
    for gt_date in args.gt_dates:
        gt_file = find_gt_file(gt_date)
        print(f"\n=== GT {gt_date} ({gt_file.name})")
        gt = build_gt(gt_file, args.spacing)
        cands = candidate_scenes(gt_date, args.max_gap)
        if not cands:
            near = closest_scene(gt_date)
            print(f"   [SALTADA] cap imatge S2 descarregada a menys de {args.max_gap} dies "
                  f"(la mes propera: {near[1]}, a {near[2]} dies). Executeu download_gt_scenes.py.")
            skipped[gt_date] = f"cap imatge S2 a menys de {args.max_gap} dies (la mes propera: {near[1]})"
            continue
        best = None
        for c_sid, c_date, c_gap in cands:      # triem l'escena que cobreix mes GT
            c_scene = load_scene(c_sid)
            c_inside, c_ok = footprint_stats(gt, c_scene)
            print(f"   candidata {c_sid}  gap {c_gap:2d} d | pixels valids {c_scene['valid'].mean():.0%} "
                  f"| GT dins la imatge {c_inside:.0%} | GT sobre pixels valids {c_ok:.0%}")
            key = (round(c_ok, 2), -c_gap)
            if best is None or key > best[0]:
                best = (key, c_sid, c_date, c_gap, c_scene)
        _, sid, s2_date, gap, scene = best
        print(f"   --> escena triada: {sid} (S2 {s2_date}, gap {gap} dies)")
        cache = {}
        _, c0 = process(scene, "none", "none", args.selection, cache)
        if metrics(signed_distances(gt, c0))["n_valid"] == 0:
            print(f"   [SALTADA] cap transecte del GT talla la costa estimada (contorns trobats: {len(c0)}).")
            print("   Causes probables: la imatge no cobreix la zona del GT (escena parcial/nodata) "
                  "o no s'ha detectat aigua.")
            skipped[gt_date] = "la imatge S2 no cobreix el GT (cobertura 0%)"
            continue
        gts[gt_date], scenes[gt_date], info[gt_date] = gt, scene, (s2_date, gap)
        for dn_name in dn_order:
            for mp_name in mp_order:
                mask, contours = process(scene, dn_name, mp_name, args.selection, cache)
                m = metrics(signed_distances(gt, contours))
                rows.append({"gt_date": gt_date, "s2_date": str(s2_date), "gap_days": gap,
                             "denoising": dn_name, "morphology": mp_name, **m})
                print(f"  {dn_name:17s} {mp_name:14s} RMSE={m['rmse']:7.2f} m  "
                      f"bias={m['bias']:7.2f}  cobertura={m['coverage']:.0%}")

    if not rows:
        print("\nCap data s'ha pogut avaluar:")
        for d, why in skipped.items():
            print(f"  - {d}: {why}")
        return
    res = pd.DataFrame(rows)
    base = res[(res.denoising == "none") & (res.morphology == "none")].set_index("gt_date").rmse
    res["delta_vs_baseline"] = res.rmse - res.gt_date.map(base)     # >0 = pitjor que sense res
    res.to_csv(OUT_DIR / "experiments_results.csv", index=False)

    # ---- resum per configuracio
    piv = res.pivot_table(index=["denoising", "morphology"], columns="gt_date", values="rmse")
    dates = [d for d in args.gt_dates if d in gts]
    piv["mean_rmse"] = piv[dates].mean(axis=1)
    piv["max_rmse"] = piv[dates].max(axis=1)
    piv["dates_under_50m"] = (piv[dates] < RMSE_TARGET).sum(axis=1)
    piv["delta_mean_vs_baseline"] = piv["mean_rmse"] - piv.loc[("none", "none"), "mean_rmse"]
    piv = piv.sort_values(["dates_under_50m", "mean_rmse"], ascending=[False, True])
    piv.round(2).to_csv(OUT_DIR / "experiments_summary.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80 + "\nTOP 10 configuracions (millor primer)\n" + "=" * 80)
    print(piv.head(10).round(2).to_string())
    print("\nBASELINE (none + none):")
    print(piv.loc[[("none", "none")]].round(2).to_string())
    print("\nRMSE mitja per denoising (promig sobre totes les morfologies):")
    print(res.groupby("denoising").rmse.mean().round(2).to_string())
    print("\nRMSE mitja per morfologia (promig sobre tots els denoising):")
    print(res.groupby("morphology").rmse.mean().round(2).to_string())

    # ---- taula en el format del enunciat (baseline vs millor)
    best_key = piv.index[0]
    print("\nTaula del enunciat (baseline  |  millor = %s + %s):" % best_key)
    print(f"{'GT date':12s}{'S2 date':12s}{'Gap (d)':>8s}{'RMSE base':>11s}{'RMSE millor':>13s}")
    for d in dates:
        r_best = res[(res.gt_date == d) & (res.denoising == best_key[0]) & (res.morphology == best_key[1])].rmse.iloc[0]
        print(f"{d:12s}{str(info[d][0]):12s}{info[d][1]:>8d}{base[d]:>11.2f}{r_best:>13.2f}")

    if not args.no_plots:
        plot_heatmap(res, dn_order, mp_order, OUT_DIR / "rmse_heatmap.png")
        mean_by = res.groupby(["denoising", "morphology"]).rmse.mean()
        only_dn = mean_by.loc[[(d, "none") for d in dn_order]]
        only_mp = mean_by.loc[[("none", m) for m in mp_order]]
        configs = [("baseline", ("none", "none")),
                   ("millor denoising", only_dn.idxmin()),
                   ("millor morfologia", only_mp.idxmin()),
                   ("millor global", best_key)]
        for d in dates:
            plot_overlay(d, gts[d], scenes[d], configs, res, args.selection,
                         OUT_DIR / f"qualitative_{d}.png")
    if skipped:
        print("\nDates NO avaluades (no surten a la taula):")
        for d, why in skipped.items():
            print(f"  - {d}: {why}")
    print(f"\nResultats a: {OUT_DIR}")


if __name__ == "__main__":
    main()