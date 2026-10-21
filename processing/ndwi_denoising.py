import time
import json
import threading
from pathlib import Path

# IMPORTANT: Forçar el backend no interactiu abans de carregar pyplot
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

# Llibreries per als filtres de denoising
from scipy.ndimage import uniform_filter
from skimage.restoration import denoise_bilateral

# ==========================================
# 1. CONFIGURACIÓ PRINCIPAL
# ==========================================
NUM_THREADS = 4  # Fils per a la descàrrega (I/O-bound)
NUM_PROCESSES = 2  # Processos per al càlcul de NDWI (CPU-bound)

# TRIA EL MÈTODE DE DENOISING: "none", "boxcar_4x4" o "bilateral"
CURRENT_DENOISING_METHOD = "bilateral"

# Directoris del projecte (Dinàmic: s'adapta automàticament a qualsevol ordinador)
base_dir = Path(__file__).resolve().parent.parent

# A partir d'aquí, definim on estan les coses respecte a la carpeta base
bands_dir = base_dir / "data" / "images_mt_1"

# La carpeta de resultats s'adapta automàticament al mètode triat per no sobreescriure res
output_dir = base_dir / "data" / f"ndwi_results_{CURRENT_DENOISING_METHOD}"
photos_dir = base_dir / "Photos"

bands_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
photos_dir.mkdir(parents=True, exist_ok=True)

geojson_path = base_dir / "data" / "Map_geojson" / "Castelldefels_map.geojson"

gdal_lock = threading.Lock()
bandes_requerides = ["red", "green", "blue", "nir"]

# Rangs temporals centrats en les dates exactes del Ground Truth + 2025
DATE_RANGES = [
    "2015-07-20/2015-08-20",  # OrtoAMBG 2015-08-05
    "2016-06-25/2016-07-25",  # OrtoAMBG 2016-07-10
    "2016-09-05/2016-10-05",  # OrtoICGC 2016-09-19
    "2017-05-10/2017-06-10",  # OrtoICGC 2017-05-25 (GT 2017-05-23)
    "2017-08-10/2017-09-10",  # OrtoAMBG 2017-08-25
    "2018-06-18/2018-07-18",  # OrtoICGC 2018-07-03
    "2019-05-08/2019-06-08",  # OrtoICGC 2019-05-23 (GT 2019-05-23)
    "2020-05-20/2020-06-20",  # OrtoICGC 2020-06-05
    "2021-05-25/2021-06-25",  # Añadido para cubrir Ground Truth 2021-06-11
    "2022-09-01/2022-10-01",  # OrtoAMBG 2022-09-17
    "2025-01-01/2025-07-01",  # Imatges actuals 2025
]


# ==========================================
# 2. FUNCIÓ DE CÀLCUL NDWI (MULTIPROCESSING)
# ==========================================
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

        # ==========================================
        # APLICACIÓ DEL FILTRE DE DENOISING SEGONS CONFIG
        # ==========================================
        if CURRENT_DENOISING_METHOD == "boxcar_4x4":
            green = uniform_filter(green, size=4, mode="reflect")
            nir = uniform_filter(nir, size=4, mode="reflect")
        elif CURRENT_DENOISING_METHOD == "bilateral":
            green = denoise_bilateral(green, sigma_color=0.1, sigma_spatial=2, channel_axis=None)
            nir = denoise_bilateral(nir, sigma_color=0.1, sigma_spatial=2, channel_axis=None)
        elif CURRENT_DENOISING_METHOD == "none":
            # No s'aplica cap filtre (passa directe amb les dades originals)
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
        plt.title(f"NDWI ({CURRENT_DENOISING_METHOD}) - {scene_id}")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        elapsed = time.time() - start_time
        return f"✔ Processat NDWI ({CURRENT_DENOISING_METHOD}): {scene_id} | Temps: {elapsed:.2f} s"
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
    print(f"PAS 2: CÀLCUL NDWI AMB MULTIPROCESSING ({NUM_PROCESSES} processos) | Mètode: {CURRENT_DENOISING_METHOD}")
    print("========================================")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(calculate_ndwi_mp, valid_scene_ids)

    for res in results:
        print(res)

    total_time = time.time() - total_start

    print("\n========================================")
    print("RESUM GLOBAL DEL PIPELINE")
    print("========================================")
    print(f"Mètode de filtratge utilitzat: {CURRENT_DENOISING_METHOD}")
    print(f"Imatges processades: {len(valid_scene_ids)}")
    print(f"Temps total: {total_time:.2f} s ({total_time / 60:.2f} min)")