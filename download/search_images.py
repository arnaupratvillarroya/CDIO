from pathlib import Path
import os
import time
import geopandas as gpd
import requests
from pystac_client import Client

# Definir la ruta base del proyecto de forma dinámica (sube 1 nivel desde download/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 1. Cargar AOI desde la carpeta de entradas del proyecto usando ruta dinámica
aoi_path = ROOT_DIR / "data" / "Map_geojson" / "polygon.geojson"

if not aoi_path.exists():
    # Fallback por si el geojson está en la ruta clásica del proyecto
    aoi_path = ROOT_DIR / "data" / "Map_geojson" / "Castelldefels_map.geojson"

aoi = gpd.read_file(aoi_path)
aoi_geometry = aoi.geometry.iloc[0].__geo_interface__

# 2. Conectarnos al STAC
catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

# 3. Buscar imágenes
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=aoi_geometry,
    datetime="2025-01-01/2025-07-01"
)

items = list(search.items())

print(f"Imágenes encontradas: {len(items)}")

# SOLO descargaremos las primeras 10 (o las que haya disponibles si son menos)
items_to_download = items[:10]

print(f"Imágenes a descargar: {len(items_to_download)}")

# 4. Crear carpeta de imágenes de entrada dentro de data/input/
output_dir = ROOT_DIR / "data" / "input" / "images"
output_dir.mkdir(parents=True, exist_ok=True)

# 5. Bandas
bands = ["red", "green", "blue", "nir"]

# Variables para las estadísticas
total_size = 0
image_times = []

# 6. Empezar cronómetro global
total_start = time.time()

# 7. Descargar las imágenes
for i, item in enumerate(items_to_download, start=1):

    print(f"\n==============================")
    print(f"Imagen {i}/{len(items_to_download)}: {item.id}")
    print(f"==============================")

    image_start = time.time()
    image_size = 0

    for band in bands:
        asset = item.assets[band]
        url = asset.href

        output = output_dir / f"{item.id}_{band}.tif"

        print(f"Descargando {band}...")

        response = requests.get(url)
        response.raise_for_status()

        with open(output, "wb") as file:
            file.write(response.content)

        size = output.stat().st_size
        image_size += size

        print(f"Guardado: {output.name}")

    image_end = time.time()
    image_time = image_end - image_start

    total_size += image_size
    image_times.append(image_time)

    print(f"Tiempo imagen: {image_time:.2f} segundos")
    print(f"Tamaño imagen: {image_size / (1024**3):.2f} GB")


# 8. Tiempo total
total_end = time.time()
total_time = total_end - total_start

# 9. Estadísticas
average_time = sum(image_times) / len(image_times) if image_times else 0
average_size = total_size / len(items_to_download) if items_to_download else 0
speed_bytes_per_second = total_size / total_time if total_time > 0 else 0
speed_mbps = (speed_bytes_per_second * 8) / (1024**2)

estimated_50_time = average_time * 50
estimated_50_size = average_size * 50

# 10. Mostrar resultados
print("\n\n")
print("========================================")
print("RESULTADOS DEL BENCHMARK DE DESCARGA")
print("========================================")

print(f"Número de imágenes: {len(items_to_download)}")
print(f"Tiempo total: {total_time:.2f} segundos ({total_time / 60:.2f} minutos)")
print(f"Tiempo medio por imagen: {average_time:.2f} segundos")
print(f"Tamaño total: {total_size / (1024**3):.2f} GB")
print(f"Tamaño medio por imagen: {average_size / (1024**3):.2f} GB")
print(f"Velocidad media: {speed_mbps:.2f} Mbps")

print("\n--- EXTRAPOLACIÓN A 50 IMÁGENES ---")
print(f"Tamaño estimado: {estimated_50_size / (1024**3):.2f} GB")
print(f"Tiempo estimado: {estimated_50_time / 60:.2f} minutos ({estimated_50_time / 3600:.2f} horas)")