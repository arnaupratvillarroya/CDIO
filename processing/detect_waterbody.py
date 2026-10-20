from pathlib import Path
import argparse
import numpy as np
import rasterio

# Definir la ruta base del proyecto (sube 2 niveles desde processing/)
BASE_DIR = Path(__file__).resolve().parent.parent


def detect_waterbody(ndwi):
    """
    Convert an NDWI array into a waterbody mask.

    NDWI > 0  -> water (True / 1)
    NDWI <= 0 -> land  (False / 0)
    """
    return ndwi > 0


def process_ndwi_file(ndwi_path, output_dir):
    """Read one NDWI GeoTIFF and save its waterbody mask."""
    output_path = output_dir / ndwi_path.name.replace(
        "_NDWI.tif", "_waterbody.tif"
    )

    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1)
        profile = src.profile.copy()

    waterbody = detect_waterbody(ndwi)

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(waterbody.astype(np.uint8), 1)

    return output_path


def test_detect_waterbody():
    """Test the NDWI -> waterbody conversion."""
    ndwi = np.array([
        [0.5, -0.2],
        [0.1, 0.0]
    ])

    expected = np.array([
        [True, False],
        [True, False]
    ])

    result = detect_waterbody(ndwi)

    assert result.dtype == bool
    assert np.array_equal(result, expected)

    print("test_detect_waterbody: OK")


if __name__ == "__main__":
    test_detect_waterbody()

    # Configurar argumentos para recibir el método por línea de comandos
    parser = argparse.ArgumentParser(description="Detección de masas de agua a partir de NDWI.")
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "boxcar_4x4", "bilateral"],
        help="Método de denoising utilizado en el NDWI."
    )
    args = parser.parse_args()
    current_method = args.method

    # Definir rutas basadas en la estructura Results y el método seleccionado
    INPUT_DIR = BASE_DIR / "data" / "Results" / "Ndwi results" / f"ndwi_results_{current_method}"
    OUTPUT_DIR = BASE_DIR / "data" / "Results" / "Waterbody results" / f"waterbody_results_{current_method}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[AVIS] No se encuentra la carpeta de entrada: {INPUT_DIR}")
        exit()

    ndwi_files = sorted(INPUT_DIR.glob("*_NDWI.tif"))

    print(f"NDWI encontrados en '{current_method}': {len(ndwi_files)}")

    for ndwi_path in ndwi_files:
        output_path = process_ndwi_file(ndwi_path, OUTPUT_DIR)
        print(f"Generado: {output_path}")

    print("Proceso completado.")