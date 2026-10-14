from pathlib import Path

import numpy as np
import rasterio


INPUT_DIR = Path("data/ndwi_results")
OUTPUT_DIR = Path("data/waterbody_results")


def detect_waterbody(ndwi):
    """
    Convert an NDWI array into a waterbody mask.

    NDWI > 0  -> water (True / 1)
    NDWI <= 0 -> land  (False / 0)
    """
    return ndwi > 0


def process_ndwi_file(ndwi_path):
    """Read one NDWI GeoTIFF and save its waterbody mask."""
    output_path = OUTPUT_DIR / ndwi_path.name.replace(
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

    # Positive NDWI -> water
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ndwi_files = sorted(INPUT_DIR.glob("*_NDWI.tif"))

    print(f"NDWI encontrados: {len(ndwi_files)}")

    for ndwi_path in ndwi_files:
        output_path = process_ndwi_file(ndwi_path)
        print(f"Generado: {output_path}")

    print("Proceso completado.")
