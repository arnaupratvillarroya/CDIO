import time
import json
import threading
from pathlib import Path
import argparse

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

from scipy.ndimage import uniform_filter
from skimage.restoration import denoise_bilateral

# ==========================================
# 1. CONFIGURACIÓ PRINCIPAL (Ruta corregida: 2 nivells amunt des de download/)
# ==========================================
NUM_THREADS = 4  
NUM_PROCESSES = 2  

base_dir = Path(__file__).resolve().parent.parent

bands_dir = base_dir / "data" / "images_mt_1"
photos_dir = base_dir / "Photos"

bands_dir.mkdir(parents=True, exist_ok=True)
photos_dir.mkdir(parents=True, exist_ok=True)

geojson_path = base_dir / "data" / "Map_geojson" / "Castelldefels_map.geojson"

gdal_lock = threading.Lock()
bandes_requerides = ["red", "green", "blue", "nir"]

DATE_RANGES = [
    "2015-07-20/2015-08-20",
    "2016-06-25/2016-07-25",
    "2016-09-05/2016-10-05",
    "2017-05-10/2017-06-10",
    "2017-08-10/2017-09-10",
    "2018-06-18/2018-07-18",
    "2019-05-08/2019-06-08",
    "2020-05-20/2020-06-20",
    "2021-05-25/2021-06-25",
    "2022-09-01/2022-10-01",
    "2025-01-01/2025-07-01",
]


# ==========================================
# 2. FUNCIÓ DE CÀLCUL NDWI (MULTIPROCESSING)
# ==========================================
def calculate_ndwi_mp(scene_id, denoising_method, output_dir):
    start_time = time.time()

    green_path = bands_dir / f"{scene_id}_green.tif"
    nir_path = bands_dir / f"{scene_id}_nir.tif"

    try:
        with rasterio.open(green_path) as green_src:
            green = green_src.read(1).astype(np.float32)
            profile = green_src.profile

        with rasterio.open(nir_path) as nir_src:
            nir = nir_src.read(1).astype(np.float32)

        # Aplicació del filtre de denoising
        if denoising_method == "boxcar_4x4":
            green = uniform_filter(green, size=4, mode="reflect")
            nir = uniform_filter(nir, size=4, mode="reflect")
        elif denoising_method == "bilateral":
            green = denoise_bilateral(green, sigma_color=0.1, sigma_spatial=2, channel_axis=None)
            nir = denoise_bilateral(nir, sigma_color=0.1, sigma_spatial=2, channel_axis=None)
        elif denoising_method == "none":
            pass

        # Càlcul del NDWI
        denominator = green + nir
        denominator[denominator == 0] = 1e-8
        ndwi = (green - nir) / denominator

        # Guardar NDWI com a GeoTIFF
        output_path = output_dir / f"{scene_id}_NDWI.tif"
        profile.update(dtype="float32", count=1, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndwi.astype(np.float32), 1)

        # Guardar visualització en PNG
        png_path = output_dir / f"{scene_id}_NDWI.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
        plt.colorbar(label="NDWI")
        plt.title(f"NDWI ({denoising_method}) - {scene_id}")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        elapsed = time.time() - start_time
        return f"✔ Processat NDWI ({denoising_method}): {scene_id} | Temps: {elapsed:.2f} s"
    except Exception as e:
        return f"✖ Error a l'escena {scene_id}: {e}"


# ==========================================
# 3. FUNCIÓ DE DESCÀRREGA (MULTI-THREADING)
# ==========================================
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


# ==========================================
# 4. EXECUCIÓ PRINCIPAL
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descàrrega Sentinel-2 i càlcul NDWI amb mètodes de denoising.")
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "boxcar_4x4", "bilateral"],
        help="Mètode de denoising a aplicar."
    )
    args = parser.parse_args()
    current_method = args.method

    output_dir = base_dir / "data" / "Results" / "Ndwi results" / f"ndwi_results_{current_method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    with open(geojson_path) as f:
        aoi_geojson = json.load(f)

    if aoi_geojson.get("type") == "FeatureCollection":
        aoi_geometry = aoi_geojson["features"][0]["geometry"]
    else:
        aoi_geometry = aoi_geojson.get("geometry", aoi_geojson)

    print("Connectant a l'API STAC d'Element 84...")
    client = Client.open("https://earth-search.aws.element84.com/v1")

    items_dict = {}
    MAX_CLOUD_COVER_PERCENTAGE = 10.0

    print("Cercant imatges Sentinel-2 per a cada rang de dates...")
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
    print(f"S'han trobat {len(items)} imatges netes en total.\n")

    if not items:
        print("No s'han trobat imatges per als rangs especificats.")
        exit()

    # --- PAS 1: Descàrrega ---
    print("========================================")
    print(f"PAS 1: DESCÀRREGA AMB MULTI-THREADING ({NUM_THREADS} fils)")
    print("========================================")

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

    print(f"\nDescàrrega finalitzada. Escenes vàlides: {len(valid_scene_ids)}")

    if not valid_scene_ids:
        print("No hi ha escenes per processar.")
        exit()

    # --- PAS 2: Càlcul NDWI ---
    print("\n========================================")
    print(f"PAS 2: CÀLCUL NDWI AMB MULTIPROCESSING ({NUM_PROCESSES} processos) | Mètode: {current_method}")
    print("========================================")

    tasks = [(scene_id, current_method, output_dir) for scene_id in valid_scene_ids]
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(calculate_ndwi_mp, tasks)

    for res in results:
        print(res)

    total_time = time.time() - total_start

    print("\n========================================")
    print("RESUM GLOBAL DEL PIPELINE")
    print("========================================")
    print(f"Mètode utilitzat: {current_method}")
    print(f"Imatges processades: {len(valid_scene_ids)}")
    print(f"Temps total: {total_time:.2f} s ({total_time / 60:.2f} min)")