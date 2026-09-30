from pystac_client import Client
import geopandas as gpd
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor


NUM_THREADS = 1
NUM_IMAGES = 5
OUTPUT_DIR = "data/images_mt_1"


# -------------------------
# AOI
# -------------------------

aoi = gpd.read_file("Castelldefels_map.geojson")
aoi_geometry = aoi.geometry.iloc[0].__geo_interface__


# -------------------------
# STAC
# -------------------------

catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=aoi_geometry,
    datetime="2025-01-01/2025-07-01"
)

items = list(search.items())

print(f"Imágenes encontradas: {len(items)}")

# Solo las primeras 5 para la prueba
items = items[:NUM_IMAGES]

print(f"Imágenes utilizadas en la prueba: {len(items)}")
print(f"Threads utilizados: {NUM_THREADS}")


# -------------------------
# Carpeta de salida
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

bands = ["red", "green", "blue", "nir"]


# -------------------------
# Descarga de una imagen
# -------------------------

def download_image(item):

    image_start = time.time()
    image_size = 0

    print(f"Descargando: {item.id}")

    for band in bands:

        url = item.assets[band].href

        output = os.path.join(
            OUTPUT_DIR,
            f"{item.id}_{band}.tif"
        )

        response = requests.get(
            url,
            stream=True
        )

        response.raise_for_status()

        with open(output, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)

        image_size += os.path.getsize(output)

    image_time = time.time() - image_start

    print(
        f"Terminada: {item.id} | "
        f"Tiempo: {image_time:.2f} s | "
        f"Tamaño: {image_size / (1024**3):.2f} GB"
    )

    return image_time, image_size


# -------------------------
# Ejecución
# -------------------------

total_start = time.time()

with ThreadPoolExecutor(
    max_workers=NUM_THREADS
) as executor:

    results = list(
        executor.map(download_image, items)
    )


# -------------------------
# Resultados
# -------------------------

total_time = time.time() - total_start

total_size = sum(
    result[1] for result in results
)

average_time = (
    sum(result[0] for result in results)
    / len(results)
)

speed_mbps = (
    (total_size / total_time)
    * 8
    / (1024 ** 2)
)


print("\n================================")
print("RESULTADOS")
print("================================")

print(f"Threads: {NUM_THREADS}")
print(f"Imágenes utilizadas: {len(items)}")
print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Tiempo total: {total_time / 60:.2f} minutos")
print(f"Tamaño total: {total_size / (1024**3):.2f} GB")
print(
    f"Tamaño medio por imagen: "
    f"{(total_size / len(items)) / (1024**3):.2f} GB"
)
print(
    f"Tiempo medio por imagen: "
    f"{average_time:.2f} segundos"
)
print(f"Velocidad media: {speed_mbps:.2f} Mbps")


# -------------------------
# Extrapolación a 50 imágenes
# -------------------------

estimated_time_50 = (
    total_time / len(items)
) * 50

estimated_size_50 = (
    total_size / len(items)
) * 50

print("\n================================")
print("EXTRAPOLACIÓN A 50 IMÁGENES")
print("================================")

print(
    f"Tiempo estimado: "
    f"{estimated_time_50 / 60:.2f} minutos"
)

print(
    f"Tamaño estimado: "
    f"{estimated_size_50 / (1024**3):.2f} GB"
)