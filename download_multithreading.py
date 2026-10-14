import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pystac_client import Client
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom

# -------------------------
# CONFIGURACIÓ PRINCIPAL
# -------------------------
NUM_THREADS = 4
OUTPUT_DIR = Path("data/images_mt_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEOJSON_PATH = "data/Castelldefels_map.geojson"
BANDS = ["red", "green", "blue", "nir"]
MAX_CLOUD_COVER = 20.0

# Rangs temporals per a cada Ground Truth + imatges actuals 2025
DATE_RANGES = [
    "2015-07-20/2015-08-20",  # GT 2015-08-05
    "2016-06-25/2016-07-25",  # GT 2016-07-10
    "2016-09-05/2016-10-05",  # GT 2016-09-19
    "2017-05-10/2017-06-10",  # GT 2017-05-25
    "2017-08-10/2017-09-10",  # GT 2017-08-25
    "2018-06-18/2018-07-18",  # GT 2018-07-03
    "2019-05-08/2019-06-08",  # GT 2019-05-23
    "2020-05-20/2020-06-20",  # GT 2020-06-05
    "2022-09-01/2022-10-01",  # GT 2022-09-17
    "2025-01-01/2025-07-01",  # Imatges actuals 2025
]

# -------------------------
# CARREGAR AOI (POLÍGON)
# -------------------------
aoi = gpd.read_file(GEOJSON_PATH)
aoi_geometry = aoi.geometry.iloc[0].__geo_interface__

# -------------------------
# CERCA EN L'API STAC
# -------------------------
print("Connectant a l'API STAC d'Element 84...")
catalog = Client.open("https://earth-search.aws.element84.com/v1")

selected_items = []

print("Cercant LA MILLOR imatge (menys núvols) per a cada data de Ground Truth...")
for date_range in DATE_RANGES:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geometry,
        datetime=date_range,
    )
    found_items = list(search.items())

    # Filtrar per percentatge de núvols
    clean_items = [
        item for item in found_items
        if item.properties.get("eo:cloud_cover", 100.0) <= MAX_CLOUD_COVER
    ]

    # Seleccionar només la imatge amb menys núvols del rang
    if clean_items:
        best_item = min(clean_items, key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
        selected_items.append(best_item)
        date_str = best_item.datetime.strftime("%Y-%m-%d")
        clouds = best_item.properties.get("eo:cloud_cover", 0)
        print(f"  [+] {date_range} -> {best_item.id} ({date_str}) | Núvols: {clouds:.1f}%")

print(f"\nTotal d'imatges seleccionades per descarregar: {len(selected_items)}\n")


# -------------------------
# DESCÀRREGA RETALLADA EN ENTRADA/EIXIDA (STREAMING)
# -------------------------
def download_cropped_image(item):
    image_start = time.time()
    scene_id = item.id
    date_str = item.datetime.strftime("%Y-%m-%d")

    # Comprovar si totes les bandes ja estan descarregades
    if all((OUTPUT_DIR / f"{scene_id}_{band}.tif").exists() for band in BANDS):
        print(f"[JA EXISTEIX] Escena {scene_id} ({date_str})")
        return 0, 0

    print(f"Descarregant i retallant AOI: {scene_id} ({date_str})")
    total_bytes = 0

    try:
        for band in BANDS:
            url = item.assets[band].href
            output_path = OUTPUT_DIR / f"{scene_id}_{band}.tif"

            # Obrir directament des del servidor HTTP i llegir NOMÉS l'àrea del polígon
            with rasterio.open(url) as src:
                aoi_projected = transform_geom('EPSG:4326', src.crs, aoi_geometry)
                crop, out_transform = mask(src, [aoi_projected], crop=True)

                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=crop.shape[1],
                    width=crop.shape[2],
                    transform=out_transform,
                    count=1,
                    dtype=crop.dtype,
                    compress="lzw"
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(crop[0], 1)

            total_bytes += output_path.stat().st_size

        image_time = time.time() - image_start
        print(
            f"✔ Completada: {scene_id} | "
            f"Temps: {image_time:.2f} s | "
            f"Mida retallada: {total_bytes / (1024 ** 2):.2f} MB"
        )
        return image_time, total_bytes

    except Exception as e:
        print(f"✖ Error descarregant {scene_id}: {e}")
        return 0, 0


# -------------------------
# EXECUCIÓ AMB MULTI-THREADING
# -------------------------
total_start = time.time()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    results = list(executor.map(download_cropped_image, selected_items))

# -------------------------
# RESULTATS
# -------------------------
total_time = time.time() - total_start
total_size = sum(r[1] for r in results)

print("\n================================")
print("RESULTATS DE LA DESCÀRREGA")
print("================================")
print(f"Fils (threads): {NUM_THREADS}")
print(f"Imatges processades: {len(selected_items)}")
print(f"Temps total: {total_time:.2f} segons ({total_time / 60:.2f} minuts)")
print(f"Mida total descarregada: {total_size / (1024 ** 2):.2f} MB")