import sys
from pathlib import Path
import numpy as np

# Ajustar sys.path para poder importar desde la carpeta processing/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "processing"))

from detect_waterbody import detect_waterbody

def test_detect_waterbody():
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
    print("Test unitari de detect_waterbody superat correctament!")

if __name__ == "__main__":
    test_detect_waterbody()