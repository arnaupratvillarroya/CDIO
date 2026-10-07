import time
import json
import threading
from pathlib import Path

# Forzar backend no interactivo antes de matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS RELATIVAS (PORTÁTIL)
# ==========================================
NUM_THREADS = 4     # Hilos para descarga (I/O)
NUM_PROCESSES = 2   # Procesos para cálculo NDWI (CPU)

# Ruta relativa automática a la ubicación actual del proyecto
base_dir = Path(__file__).resolve().parent
bands_dir = base_dir / "data" / "images_mt_1"
output_dir = base_dir / "data" / "ndwi_results"
photos_dir = base_dir / "Photos"

bands_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
photos_dir.mkdir(parents=True, exist_ok=True)

geojson_path = base_dir / "Castelldefels_map.geojson"

gdal_lock = threading.Lock()
bandes_requerides = ["red", "green", "blue", "nir"]

# Rango de fechas que cubre las 3 fechas de Ground Truth + 2025 H1
DATE_RANGES = [
    "2015-07-20/2015-08-20",
    "2016-06-25/2016-07-25",
    "2016-09-05/2016-10-05",
    "2017-05-10/2017-06-10",  # GT 2017-05-23
    "2018-06-18/2018-07-18",
    "2019-05-08/2019-06-08",  # GT 2019-05-23
    "2020-05-20/2020-06-20",
    "2021-05-25/2021-06-25",  # GT 2021-06-11 (AÑADIDO)
    "2022-09-01/2022-10-01",
    "2025-01-01/2025-07-01",  # Imágenes 2025 H1
]


def calculate_ndwi_mp(scene_id):
    start_time = time.time()
    green_path = bands_dir / f"{scene_id}_green.tif"
    nir_path = bands_dir / f"{scene_id}_nir.tif"

    try:
        with rasterio.open(green_path) as green_src:
            green = green_src.read(1).astype(np.float32)
            profile = green_src.profile

        with rasterio.open(nir_path) as nir_src:
            nir = nir_src.read(1).astype(np.float32)

        denominator = green + nir
        denominator[denominator == 0] = 1e-8
        ndwi = (green - nir) / denominator

        output_path = output_dir / f"{scene_id}_NDWI.tif"
        profile.update(dtype="float32", count=1, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndwi.astype(np.float32), 1)

        png_path = output_dir / f"{scene_id}_NDWI.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
        plt.colorbar(label="NDWI")
        plt.title(f"NDWI - {scene_id}")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        elapsed = time.time() - start_time
        return f"✔ Processat NDWI: {scene_id} | Temps: {elapsed:.2f} s"
    except Exception as e:
        return f"✖ Error a l'escena {scene_id}: {e}"


def download_single_scene(item, index, total_items, aoi_geometry):
    scene_id = item.id
    date_str = item.datetime.strftime("%Y-%m-%d")

    green_path = bands_dir / f"{scene_id}_green.tif"
    nir_path = bands_dir / f"{scene_id}_nir.tif"

    if green_path.exists() and nir_path.exists():
        print(f"[{index}/{total_items}] Escena {scene_id} ({date_str}) ja existeix a disc.")
        return scene_id

    assets = item.assets
    if not all(banda in assets for banda in bandes_requerides):
        return None

    try:
        print(f"[{index}/{total_items}] Descarregant: {scene_id} ({date_str})")
        bands_data = {}
        aoi_projectat = None

        for band in bandes_requerides:
            url = assets[band].href
            with gdal_lock:
                with rasterio.open(url) as src:
                    if aoi_projectat is None:
                        aoi_projectat = transform_geom('EPSG:4326', src.crs, aoi_geometry)
                    crop, out_transform = mask(src, [aoi_projectat], crop=True)

                    prof = src.profile.copy()
                    prof.update(
                        driver="GTiff",
                        height=crop.shape[1],
                        width=crop.shape[2],
                        transform=out_transform,
                        count=1,
                        dtype=crop.dtype
                    )
            bands_data[band] = (crop[0], prof)

        for band in bandes_requerides:
            arr, prof = bands_data[band]
            if np.all(arr == 0):
                return None

            out_band_path = bands_dir / f"{scene_id}_{band}.tif"
            with rasterio.open(out_band_path, "w", **prof) as dst:
                dst.write(arr, 1)

        return scene_id
    except Exception as e:
        print(f"  Error descarregant {scene_id}: {e}")
        return None


if __name__ == "__main__":
    total_start = time.time()

    with open(geojson_path) as f:
        aoi_geojson = json.load(f)

    if aoi_geojson.get("type") == "FeatureCollection":
        aoi_geometry = aoi_geojson["features"][0]["geometry"]
    else:
        aoi_geometry = aoi_geojson.get("geometry", aoi_geojson)

    client = Client.open("https://earth-search.aws.element84.com/v1")

    items_dict = {}
    MAX_CLOUD_COVER_PERCENTAGE = 10.0

    print("Cercant imatges Sentinel-2...")
    for date_range in DATE_RANGES:
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=aoi_geometry,
            datetime=date_range,
        )
        for item in search.items():
            if item.properties.get("eo:cloud_cover", 100.0) < MAX_CLOUD_COVER_PERCENTAGE:
                items_dict[item.id] = item

    items = list(items_dict.values())
    print(f"Imatges trobades: {len(items)}\n")

    if not items:
        print("No s'han trobat imatges.")
        exit()

    print("PAS 1: Descàrrega...")
    valid_scene_ids = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(download_single_scene, item, i, len(items), aoi_geometry): item
            for i, item in enumerate(items, 1)
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                valid_scene_ids.append(res)

    print("\nPAS 2: Càlcul NDWI...")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(calculate_ndwi_mp, valid_scene_ids)

    for res in results:
        print(res)

    total_time = time.time() - total_start
    print(f"\nFinalitzat en {total_time / 60:.2f} minuts.")
