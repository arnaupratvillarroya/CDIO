from pathlib import Path
import argparse
import numpy as np
import rasterio
from scipy.ndimage import convolve

# Definir la ruta base del proyecto (sube 2 niveles desde processing/)
BASE_DIR = Path(__file__).resolve().parent.parent


def detect_coastline(waterbody):
    """
    Detect coastline pixels using an 8-neighbor morphological approach.

    A coastline pixel is a water pixel that has at least one
    neighboring land pixel.
    """

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    neighbor_sum = convolve(
        waterbody.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )

    coastline = np.logical_and(
        waterbody,
        neighbor_sum < 8
    )

    return coastline


def process_waterbody_file(waterbody_path, output_dir):
    """Read one waterbody GeoTIFF and save its coastline."""
    
    output_path = output_dir / waterbody_path.name.replace(
        "_waterbody.tif",
        "_coastline.tif"
    )

    with rasterio.open(waterbody_path) as src:
        waterbody = src.read(1).astype(bool)
        profile = src.profile.copy()

    coastline = detect_coastline(waterbody)

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(coastline.astype(np.uint8), 1)

    return output_path


def test_detect_coastline():
    """Test the coastline detection function."""

    waterbody = np.array([
        [False, False, False, False],
        [False, True,  True,  False],
        [False, True,  True,  False],
        [False, False, False, False]
    ])

    coastline = detect_coastline(waterbody)

    expected = np.array([
        [False, False, False, False],
        [False, True,  True,  False],
        [False, True,  True,  False],
        [False, False, False, False]
    ])

    assert coastline.dtype == bool
    assert np.array_equal(coastline, expected)

    print("test_detect_coastline: OK")


if __name__ == "__main__":
    test_detect_coastline()

    # Configurar argumentos para recibir el método por línea de comandos
    parser = argparse.ArgumentParser(description="Extracción de línea de costa a partir de máscaras de agua.")
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "boxcar_4x4", "bilateral"],
        help="Método de denoising utilizado."
    )
    args = parser.parse_args()
    current_method = args.method

    # Definir rutas dinámicas basadas en la estructura Results y el método seleccionado
    INPUT_DIR = BASE_DIR / "data" / "Results" / "Waterbody results" / f"waterbody_results_{current_method}"
    OUTPUT_DIR = BASE_DIR / "data" / "Results" / "Coastline results" / f"coastline_results_{current_method}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[AVIS] No se encuentra la carpeta de entrada: {INPUT_DIR}")
        exit()

    waterbody_files = sorted(
        INPUT_DIR.glob("*_waterbody.tif")
    )

    print(f"Waterbody encontrados en '{current_method}': {len(waterbody_files)}")

    for waterbody_path in waterbody_files:
        output_path = process_waterbody_file(waterbody_path, OUTPUT_DIR)
        print(f"Generado: {output_path}")

    print("Proceso completado.")