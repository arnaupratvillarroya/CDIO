import datetime
import re
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import KDTree
from shapely.geometry import LineString

# ==========================================
# 1. VISUALITZACIÓ I INTERACCIÓ AMB GEOTIFF
# ==========================================
def show_geotiff(tif_path, downsample_normalize=True, title=""):
    """Visualitza un GeoTIFF i permet fer clic per veure les coordenades del píxel."""
    with rasterio.open(tif_path) as src:
        img = src.read(1)
        transform = src.transform

        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap="gray")
        plt.colorbar(im)
        plt.title(title)

        clicked_coords = []

        def onclick(event):
            if event.inaxes != ax:
                return
            x_pix, y_pix = int(event.xdata), int(event.ydata)
            val = img[y_pix, x_pix]

            # Valor mitjà en els 8 píxels del voltant (finestra 3x3)
            avg_val = np.mean(
                img[max(0, y_pix - 1) : y_pix + 2, max(0, x_pix - 1) : x_pix + 2]
            )

            lon, lat = rasterio.transform.xy(transform, y_pix, x_pix)
            print(f"Clicked pixel: ({x_pix}, {y_pix})")
            print(f"Coordinates: ({lon:.6f}, {lat:.6f})")
            print(f"Pixel Value: {val}")
            print(f"Average Value (3x3 window): {avg_val:.2f}")
            clicked_coords.append((lon, lat, val))
            plt.close(fig)  # Tanca després d'un clic

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

        if clicked_coords:
            return clicked_coords[0]
        else:
            return None


# ==========================================
# 2. GESTIÓ DE RUTES I DATASETS
# ==========================================
def project_path(project):
    p = Path(project).expanduser().resolve()
    if p.exists() and p.is_dir():
        return p

    # CANVIA AQUESTA LÍNIA si els projectes estan en una altra carpeta (ex: "data"):
    projects_dir = Path("data") / project
    if projects_dir.exists():
        return projects_dir.resolve()

    return p


def extract_date(filename):
    """Extreu una data (format YYYY-MM-DD o YYYY_MM_DD) del nom d'un fitxer."""
    filename_str = str(filename)

    # Cerca formats YYYY-MM-DD o YYYY_MM_DD
    match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", filename_str)
    if match:
        y, m, d = match.groups()
        return datetime.date(int(y), int(m), int(d))

    # Cerca format compacte YYYYMMDD
    match_compact = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", filename_str)
    if match_compact:
        y, m, d = match_compact.groups()
        return datetime.date(int(y), int(m), int(d))

    return None


def load_coords_csv(filepath):
    """Carrega coordenades (X, Y) des d'un fitxer CSV o TSV."""
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(filepath, sep="\t")

    cols_lower = {str(col).lower(): col for col in df.columns}
    if "x" in cols_lower and "y" in cols_lower:
        return df[[cols_lower["x"], cols_lower["y"]]].values

    return df.iloc[:, :2].values


def export_coords_to_kmz(coords, output_path):
    """Genera un fitxer KMZ compatible amb Google Earth a partir de les coordenades."""
    output_path = Path(output_path)
    kml_filename = output_path.stem + ".kml"

    coord_str = "\n".join([f"{c[0]},{c[1]},0" for c in coords])
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{output_path.stem}</name>
    <Placemark>
      <name>Shoreline Reference</name>
      <LineString>
        <coordinates>
{coord_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr(kml_filename, kml_content)


# ==========================================
# 3. CONSTRUCCIÓ I ORDENACIÓ DE LA LÍNIA DE COSTA
# ==========================================
def build_ordered_linestring(coords, is_ordered=False):
    """Construeix una LineString de Shapely.

    Si is_ordered=False, ordena espacialment els punts seguint el veí més
    proper.
    """
    coords = np.asarray(coords)
    if len(coords) < 2:
        raise ValueError(
            "Es necessiten almenys 2 punts per construir una LineString."
        )

    if is_ordered:
        return LineString(coords)

    # Si els punts no estan ordenats (ex: extrets d'un raster),
    # els ordenem usant l'algorisme del veí més proper
    unvisited = list(range(len(coords)))
    start_idx = int(np.argmin(coords[:, 0]))  # Comença pel punt més a l'esquerra

    ordered_indices = [start_idx]
    unvisited.remove(start_idx)

    tree = KDTree(coords)
    curr_idx = start_idx

    while unvisited:
        # Cerca els veïns més propers
        distances, indices = tree.query(
            coords[curr_idx], k=min(10, len(coords))
        )
        if not isinstance(indices, (np.ndarray, list)):
            indices = [indices]

        next_idx = None
        for idx in indices:
            if idx in unvisited:
                next_idx = idx
                break

        if next_idx is None:
            # Si no troba veïns a la cerca local, agafa el més proper de tots els no visitats
            unvisited_coords = coords[unvisited]
            dists = np.linalg.norm(
                unvisited_coords - coords[curr_idx], axis=1
            )
            next_idx = unvisited[int(np.argmin(dists))]

        ordered_indices.append(next_idx)
        unvisited.remove(next_idx)
        curr_idx = next_idx

    ordered_coords = coords[ordered_indices]
    return LineString(ordered_coords)


# ==========================================
# EXECUCIÓ DIRECTA
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_geotiff(sys.argv[1])
    else:
        print("Ús: python shoreline_utils.py <ruta_al_fitxer_geotiff>")