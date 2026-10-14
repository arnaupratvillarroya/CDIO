from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import convolve


INPUT_DIR = Path("data/waterbody_results")
OUTPUT_DIR = Path("data/coastline_results")


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


def process_waterbody_file(waterbody_path):
    """Read one waterbody GeoTIFF and save its coastline."""
    
    output_path = OUTPUT_DIR / waterbody_path.name.replace(
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    waterbody_files = sorted(
        INPUT_DIR.glob("*_waterbody.tif")
    )

    print(f"Waterbody encontrados: {len(waterbody_files)}")

    for waterbody_path in waterbody_files:
        output_path = process_waterbody_file(waterbody_path)
        print(f"Generado: {output_path}")

    print("Proceso completado.")

