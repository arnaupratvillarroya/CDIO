import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
from pystac_client import Client
from rasterio.mask import mask
from rasterio.warp import transform_geom

BASE_DIR = Path(__file__).resolve().parent.parent
BANDS_DIR = BASE_DIR / "data" / "images_mt_1"
AOI_PATH = BASE_DIR / "data" / "Map_geojson" / "Castelldefels_map.geojson"

GT_DATES = ["2017-05-25", "2019-05-23", "2021-06-09"]
BANDS = ["green", "nir"]
COLLECTIONS = ["sentinel-2-l2a", "sentinel-2-l1c"]


def load_aoi():
    with open(AOI_PATH) as f:
        g = json.load(f)
    if g.get("type") == "FeatureCollection":
        return g["features"][0]["geometry"]
    return g.get("geometry", g)


def pick_candidates(items, target, max_cloud, k):
    """Una escena per dia (la de menys nuvols, <= max_cloud); les k mes properes a target."""
    best = {}
    for it in items:
        cloud = it.properties.get("eo:cloud_cover", 100.0)
        day = it.datetime.date()
        if cloud <= max_cloud and (day not in best or cloud < best[day].properties.get("eo:cloud_cover", 100.0)):
            best[day] = it
    ranked = sorted(best.values(), key=lambda i: (abs((i.datetime.date() - target.date()).days), i.id))
    return ranked[:k]


def find_items(catalog, aoi, gt_date, window, max_cloud, k):
    target = datetime.strptime(gt_date, "%Y-%m-%d")
    rng = f"{target - timedelta(days=window):%Y-%m-%d}/{target + timedelta(days=window):%Y-%m-%d}"
    for coll in COLLECTIONS:
        items = list(catalog.search(collections=[coll], intersects=aoi, datetime=rng).items())
        items = [i for i in items if all(b in i.assets for b in BANDS)]
        print(f"  {coll}: {len(items)} escenes a {rng}")
        if items:
            return pick_candidates(items, target, max_cloud, k), target
    return [], target


def download(item, aoi):
    sid = item.id
    outs = {b: BANDS_DIR / f"{sid}_{b}.tif" for b in BANDS}
    if all(p.exists() for p in outs.values()):
        print(f"    [ja existeix] {sid}")
        return True
    try:
        aoi_proj = None
        for b in BANDS:
            with rasterio.open(item.assets[b].href) as src:
                if aoi_proj is None:
                    aoi_proj = transform_geom("EPSG:4326", src.crs, aoi)
                crop, tr = mask(src, [aoi_proj], crop=True)
                prof = src.profile.copy()
                prof.update(driver="GTiff", height=crop.shape[1], width=crop.shape[2],
                            transform=tr, count=1, dtype=crop.dtype, compress="lzw")
            if not crop.any():
                raise ValueError("retall buit (tot nodata)")
            with rasterio.open(outs[b], "w", **prof) as dst:
                dst.write(crop[0], 1)
        print(f"    [descarregada] {sid}  (pixels no nuls: {(crop[0] > 0).mean():.0%})")
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"    [ERROR] {sid}: {e}")
        for p in outs.values():
            p.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser(description="Descarrega escenes S2 properes al ground truth")
    ap.add_argument("--gt-dates", nargs="+", default=GT_DATES)
    ap.add_argument("--window", type=int, default=20, help="dies a cada costat del GT")
    ap.add_argument("--max-cloud", type=float, default=30.0, help="% maxim de nuvols de l'escena")
    ap.add_argument("-k", type=int, default=3, help="escenes (dies) a descarregar per data de GT")
    ap.add_argument("--list", action="store_true", help="nomes mostra els candidats")
    args = ap.parse_args()

    BANDS_DIR.mkdir(parents=True, exist_ok=True)
    aoi = load_aoi()
    catalog = Client.open("https://earth-search.aws.element84.com/v1")

    for gt_date in args.gt_dates:
        print(f"\n=== GT {gt_date}")
        items, target = find_items(catalog, aoi, gt_date, args.window, args.max_cloud, args.k)
        if not items:
            print(f"  [AVIS] cap escena amb <= {args.max_cloud}% de nuvols a +-{args.window} dies "
                  f"(ni L2A ni L1C).")
            continue
        for it in items:
            gap = abs((it.datetime.date() - target.date()).days)
            cloud = it.properties.get("eo:cloud_cover", float("nan"))
            coll = it.collection_id
            print(f"  candidata {it.datetime:%Y-%m-%d} (a {gap} d)  nuvols {cloud:.1f}%  {coll}  {it.id}")
            if not args.list:
                download(it, aoi)

    print("\nFet. Ara executeu:  python experiments_morphology.py --quick")


if __name__ == "__main__":
    main()