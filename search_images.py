from pystac_client import Client
import geopandas as gpd
import os
import requests
import time


# 1. Cargar AOI
aoi = gpd.read_file("../data/Castelldefels_map.geojson")
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

# SOLO descargaremos las primeras 10
items_to_download = items[:10]

print(f"Imágenes a descargar: {len(items_to_download)}")

# 4. Crear carpeta
os.makedirs("../data/images", exist_ok=True)

# 5. Bandas
bands = ["red", "green", "blue", "nir"]

# Variables para las estadísticas
total_size = 0
image_times = []

# 6. Empezar cronómetro global
total_start = time.time()

# 7. Descargar las 10 imágenes
for i, item in enumerate(items_to_download, start=1):

    print(f"\n==============================")
    print(f"Imagen {i}/10: {item.id}")
    print(f"==============================")

    image_start = time.time()
    image_size = 0

    for band in bands:

        asset = item.assets[band]
        url = asset.href

        output = f"../data/images/{item.id}_{band}.tif"

        print(f"Descargando {band}...")

        response = requests.get(url)
        response.raise_for_status()

        with open(output, "wb") as file:
            file.write(response.content)

        size = os.path.getsize(output)

        image_size += size

        print(f"Guardado: {output}")

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
average_time = sum(image_times) / len(image_times)

average_size = total_size / len(items_to_download)

speed_bytes_per_second = total_size / total_time

speed_mbps = (speed_bytes_per_second * 8) / (1024**2)

estimated_50_time = average_time * 50

estimated_50_size = average_size * 50


# 10. Mostrar resultados
print("\n\n")
print("========================================")
print("RESULTADOS")
print("========================================")

print(f"Número de imágenes: {len(items_to_download)}")

print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Tiempo total: {total_time / 60:.2f} minutos")

print(f"Tiempo medio por imagen: {average_time:.2f} segundos")
print(f"Tiempo medio por imagen: {average_time / 60:.2f} minutos")

print(f"Tamaño total: {total_size / (1024**3):.2f} GB")
print(f"Tamaño medio por imagen: {average_size / (1024**3):.2f} GB")

print(f"Velocidad media: {speed_mbps:.2f} Mbps")

print("\n--- EXTRAPOLACIÓN A 50 IMÁGENES ---")

print(f"Tamaño estimado: {estimated_50_size / (1024**3):.2f} GB")
print(f"Tiempo estimado: {estimated_50_time / 60:.2f} minutos")
print(f"Tiempo estimado: {estimated_50_time / 3600:.2f} horas")