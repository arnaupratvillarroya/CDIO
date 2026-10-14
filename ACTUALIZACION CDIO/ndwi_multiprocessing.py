from pathlib import Path
from multiprocessing import Pool
import time
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURACIÓN
# ==========================================

IMAGES_DIR = Path("data/images_mt_1")
OUTPUT_DIR = Path("data/ndwi_results")

NUM_PROCESSES = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# PROCESAR UNA IMAGEN
# ==========================================

def calculate_ndwi(scene_id):
    start_time = time.time()

    green_path = IMAGES_DIR / f"{scene_id}_green.tif"
    nir_path = IMAGES_DIR / f"{scene_id}_nir.tif"

    print(f"Procesando: {scene_id}")

    with rasterio.open(green_path) as green_src:
        green = green_src.read(1)
        profile = green_src.profile

    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1)

    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    denominator = green + nir

    # Evitar división por cero
    denominator[denominator == 0] = 1e-8

    ndwi = (green - nir) / denominator

    # Guardar NDWI como GeoTIFF
    output_path = OUTPUT_DIR / f"{scene_id}_NDWI.tif"

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndwi.astype(np.float32), 1)

    # Guardar visualización
    png_path = OUTPUT_DIR / f"{scene_id}_NDWI.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
    plt.colorbar(label="NDWI")
    plt.title(f"NDWI - {scene_id}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time

    print(
        f"Finalizado: {scene_id} | "
        f"Tiempo: {elapsed:.2f} s"
    )

    return elapsed


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # Obtener las escenas a partir de los archivos Green
    green_files = sorted(IMAGES_DIR.glob("*_green.tif"))

    scene_ids = [
        file.name.replace("_green.tif", "")
        for file in green_files
    ]

    print("================================")
    print("NDWI CON MULTIPROCESSING")
    print("================================")
    print(f"Imágenes encontradas: {len(scene_ids)}")
    print(f"Procesos utilizados: {NUM_PROCESSES}")

    if not scene_ids:
        print("No se encontraron imágenes.")
        exit()

    total_start = time.time()

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(calculate_ndwi, scene_ids)

    total_time = time.time() - total_start

    print("\n================================")
    print("RESULTADOS")
    print("================================")
    print(f"Procesos: {NUM_PROCESSES}")
    print(f"Imágenes procesadas: {len(results)}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Tiempo total: {total_time / 60:.2f} minutos")
    print(f"Tiempo medio por imagen: {np.mean(results):.2f} segundos")

    print("\nProceso NDWI completado.")