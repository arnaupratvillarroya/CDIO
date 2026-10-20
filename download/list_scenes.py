"""
list_scenes.py
--------------
Llista les escenes Sentinel-2 L2A disponibles (amb el % de nuvols de TOTA l'escena)
al voltant de cada data de ground truth, per decidir quina descarregar.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from pystac_client import Client

BASE_DIR = Path(__file__).resolve().parent.parent
GT_DATES = ["2017-05-25", "2019-05-23", "2021-06-09"]
WINDOW_DAYS = 20

with open(BASE_DIR / "data" / "Map_geojson" / "Castelldefels_map.geojson") as f:
    aoi = json.load(f)
geom = aoi["features"][0]["geometry"] if aoi.get("type") == "FeatureCollection" else aoi.get("geometry", aoi)

catalog = Client.open("https://earth-search.aws.element84.com/v1")

for gt in GT_DATES:
    d = datetime.strptime(gt, "%Y-%m-%d")
    rng = f"{d - timedelta(days=WINDOW_DAYS):%Y-%m-%d}/{d + timedelta(days=WINDOW_DAYS):%Y-%m-%d}"
    items = sorted(
        catalog.search(collections=["sentinel-2-l2a"], intersects=geom, datetime=rng).items(),
        key=lambda i: i.datetime,
    )
    print(f"\nGT {gt}  (cerca {rng})  -> {len(items)} escenes")
    for it in items:
        gap = abs((it.datetime.replace(tzinfo=None) - d).days)
        clouds = it.properties.get("eo:cloud_cover", float("nan"))
        print(f"  {it.datetime:%Y-%m-%d}  a {gap:2d} dies del GT  nuvols {clouds:5.1f}%  {it.id}")