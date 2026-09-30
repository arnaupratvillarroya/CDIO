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

# ==========================================
# 1. CONFIGURACIÓ PRINCIPAL
# ==========================================
NUM_THREADS = 1      # Fils per a la descàrrega (I/O-bound)
NUM_PROCESSES = 1    # Processos per al càlcul de NDWI (CPU-bound)

# Directoris del projecte
base_dir = Path(r"C:\Users\Usuario\CDIO")
bands_dir = base_dir / "data" / "images_mt_1"
output_dir = base_dir / "data" / "ndwi_results"
photos_dir = base_dir / "Photos"

bands_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
photos_dir.mkdir(parents=True, exist_ok=True)

geojson_path = base_dir / "Castelldefels_map.geojson"

gdal_lock = threading.Lock()
bandes_requerides = ["red", "green", "blue", "nir"]

# ==========================================
# 2. FUNCIÓ INDEPENDENT DE MULTIPROCESSING (NDWI)
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
        plt.title(f"NDWI - {scene_id}")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        elapsed = time.time() - start_time
        return f"✔ Processat NDWI: {scene_id} | Temps: {elapsed:.2f} s"
    except Exception as e:
        return f"✖ Error a l'escena {scene_id}: {e}"


# ==========================================
# 3. FUNCIÓ DE DESCÀRREGA AMB MULTI-THREADING OPTIMITZAT
# ==========================================
def download_single_scene(item, index, total_items, aoi_geometry):
    scene_id = item.id
    date_str = item.datetime.strftime("%Y-%m-%d")
    
    green_path = bands_dir / f"{scene_id}_green.tif"
    nir_path = bands_dir / f"{scene_id}_nir.tif"
    
    if green_path.exists() and nir_path.exists():
        print(f"[{index}/{total_items}] Escena {scene_id} ja descarregada prèviament.")
        return scene_id

    assets = item.assets
    if not all(banda in assets for banda in bandes_requerides):
        return None

    try:
        print(f"[{index}/{total_items}] Descarregant bandes (fils): {scene_id} ({date_str})")
        bands_data = {}
        aoi_projectat = None

        for band in bandes_requerides:
            url = assets[band].href
            # PROTECCIÓ REDUÏDA: Només es bloqueja l'accés crític de lectura/màscara de GDAL
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

        # Guardar les bandes a disc fora del lock principal
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
# 4. EXECUCIÓ PRINCIPAL (MAIN)
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

    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geometry,
        datetime="2025-01-01/2025-07-01",
    )

    all_items = list(search.items())
    MAX_CLOUD_COVER_PERCENTAGE = 10.0
    items = [item for item in all_items if item.properties.get("eo:cloud_cover", 100.0) < MAX_CLOUD_COVER_PERCENTAGE]

    print(f"S'han trobat {len(items)} imatges netes per processar.\n")

    # --- PAS 1: Descàrrega amb Multi-threading Optimitzat ---
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

    # --- PAS 2: Càlcul de NDWI amb Multi-processing ---
    print("\n========================================")
    print(f"PAS 2: CÀLCUL NDWI AMB MULTIPROCESSING ({NUM_PROCESSES} processos)")
    print("========================================")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(calculate_ndwi_mp, valid_scene_ids)

    for res in results:
        print(res)

    total_time = time.time() - total_start

    print("\n========================================")
    print("RESUM GLOBAL DEL PIPELINE")
    print("========================================")
    print(f"Imatges processades: {len(valid_scene_ids)}")
    print(f"Temps total del pipeline: {total_time:.2f} segons ({total_time / 60:.2f} minuts)")
    print("Procés complet finalitzat amb èxit.")