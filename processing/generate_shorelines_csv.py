import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from skimage.measure import find_contours

# Definir la ruta base del proyecto de forma dinámica (sube 2 niveles desde processing/)
BASE_DIR = Path(__file__).resolve().parent.parent


def extract_date_from_name(filename):
    match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", str(filename))
    if match:
        date_str = match.group(1).replace("_", "-")
        if len(date_str) == 8 and "-" not in date_str:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str
    return "2025-01-01"


def main():
    # Configurar argumentos para recibir el método por línea de comandos
    parser = argparse.ArgumentParser(description="Generación de CSV consolidado con las líneas de costa.")
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "boxcar_4x4", "bilateral"],
        help="Método de denoising utilizado."
    )
    args = parser.parse_args()
    current_method = args.method

    # Definir rutas basadas en la estructura actual
    coastline_dir = BASE_DIR / "data" / "Results" / "Coastline results" / f"coastline_results_{current_method}"
    
    # Directorio de salida adaptado para guardar el CSV según el método
    output_dir = BASE_DIR / "data" / "output" / "estimated_waterbodies_edges"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del CSV adaptado o general (mantenemos all_shorelines.csv o por método)
    output_csv = output_dir / f"all_shorelines_{current_method}.csv"

    all_rows = []
    
    if not coastline_dir.exists():
        print(f"[AVIS] No se encuentra la carpeta de entrada: {coastline_dir}")
        return

    tif_files = sorted(coastline_dir.glob("*_coastline.tif"))

    print(f"Trobats {len(tif_files)} fitxers de línia de costa per a '{current_method}'...")

    for tif_path in tif_files:
        date_str = extract_date_from_name(tif_path.name)
        print(f"Processant {tif_path.name} (Data: {date_str})...")

        with rasterio.open(tif_path) as src:
            img = src.read(1)
            transform = src.transform

            # 1. Eliminar soroll de les vores del GeoTIFF (5 píxels de marge)
            marge = 5
            img[:marge, :] = 0
            img[-marge:, :] = 0
            img[:, :marge] = 0
            img[:, -marge:] = 0

            if np.max(img) == 0:
                continue

            # 2. Extreure contorns ordenats
            contours = find_contours(img, level=0.5)

            if not contours:
                continue

            # Agafem el contorn més llarg (la línia de costa principal)
            main_contour = max(contours, key=len)

            # 3. Convertir a coordenades geogràfiques mantenint l'ordre
            for r, c in main_contour:
                x, y = rasterio.transform.xy(transform, r, c)
                all_rows.append({"date": date_str, "x": x, "y": y})

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)
        print(f"\n✔ Fitxer CSV regenerat correctament: {output_csv}")
    else:
        print("\n✖ No s'han trobat contorns vàlids.")


if __name__ == "__main__":
    main()