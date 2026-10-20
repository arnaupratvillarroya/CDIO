import sys
from pathlib import Path
import numpy as np

# Ajustar sys.path para poder importar desde la carpeta processing/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "processing"))

from coastline_estimator import detect_coastline

def test_detect_coastline():
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
    print("Test unitari de coastline superat correctament!")

if __name__ == "__main__":
    test_detect_coastline()