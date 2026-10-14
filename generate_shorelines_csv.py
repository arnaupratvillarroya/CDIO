import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from skimage.measure import find_contours

# Rutes
base_dir = Path("/home/cbltic/Escriptori/CDIO_1/CDIO")
coastline_dir = base_dir / "data" / "coastline_results"
output_dir = (
    base_dir / "data" / "output" / "estimated_waterbodies_edges"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "all_shorelines.csv"


def extract_date_from_name(filename):
  match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", str(filename))
  if match:
    date_str = match.group(1).replace("_", "-")
    if len(date_str) == 8 and "-" not in date_str:
      date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
  return "2025-01-01"


all_rows = []
tif_files = sorted(coastline_dir.glob("*_coastline.tif"))

print(f"Trobats {len(tif_files)} fitxers de línia de costa...")

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