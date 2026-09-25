import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client

# --- 1. Definir configuració i directoris ---
photos_dir = Path("/home/cbltic/Escriptori/CDIO_1/CDIO/Photos")
photos_dir.mkdir(parents=True, exist_ok=True)

geojson_path = Path("/home/cbltic/Escriptori/CDIO_1/CDIO/aoi.geojson")

# --- 2. Carregar la geometria de l'AOI ---
with open(geojson_path) as f:
    aoi_geojson = json.load(f)

if aoi_geojson.get("type") == "FeatureCollection":
    aoi_geometry = aoi_geojson["features"][0]["geometry"]
else:
    aoi_geometry = aoi_geojson.get("geometry", aoi_geojson)

# --- 3. Connectar a l'API STAC i cercar les imatges ---
print("Connectant a l'API STAC d'Element 84...")
stac_url = "https://earth-search.aws.element84.com/v1"
client = Client.open(stac_url)

search = client.search(
    collections=["sentinel-2-l2a"],
    intersects=aoi_geometry,
    datetime="2025-01-01/2025-07-01",
)

items = list(search.items())
print(f"S'han trobat {len(items)} imatges coincidentes amb els criteris.\n")

bandes_requerides = ["red", "green", "blue", "nir"]

# --- 4. Iterar sobre totes les N imatges ---
for i, item in enumerate(items, 1):
    scene_id = item.id
    date_str = item.datetime.strftime("%Y-%m-%d")
    print(f"[{i}/{len(items)}] Processant escena: {scene_id} | Data: {date_str}")

    assets = item.assets
    if not all(banda in assets for banda in bandes_requerides):
        print(f"  AVÍS: Falten bandes en aquesta escena. S'omet.")
        continue

    try:
        # --- 5. Descarregar i retallar les 4 bandes ---
        bands_data = {}
        aoi_projectat = None

        for band in bandes_requerides:
            url = assets[band].href
            with rasterio.open(url) as src:
                if aoi_projectat is None:
                    # Reprojectem el polígon (EPSG:4326) al sistema de coordenades de la imatge (UTM 31N)
                    aoi_projectat = transform_geom('EPSG:4326', src.crs, aoi_geometry)

                crop, _ = mask(src, [aoi_projectat], crop=True)
                bands_data[band] = crop[0]

        red = bands_data["red"]
        green = bands_data["green"]
        blue = bands_data["blue"]
        nir = bands_data["nir"]

        # --- 6. Calcular el NDWI ---
        green_f = green.astype(float)
        nir_f = nir.astype(float)

        denominador = green_f + nir_f
        denominador[denominador == 0] = 1e-8
        ndwi = (green_f - nir_f) / denominador

        # --- 7. Crear composició a Color Real (RGB) ---
        rgb = np.dstack((red, green, blue)) / 3000.0
        rgb = np.clip(rgb, 0, 1)

        # --- 8. Visualització en 2 files i 3 columnes amb contrast ajustat ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Sentinel-2 L2A (Castelldefels) - Data: {date_str}\nID: {scene_id}", fontsize=14)

        # Fixem el màxim a 3000 per evitar que la imatge es vegi descolorida
        v_max_bandes = 3000

        # Fila 1: Bandes Red, Green, Blue
        im0 = axes[0, 0].imshow(red, cmap="Reds", vmin=0, vmax=v_max_bandes)
        axes[0, 0].set_title("Banda Red (B04)")
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(green, cmap="Greens", vmin=0, vmax=v_max_bandes)
        axes[0, 1].set_title("Banda Green (B03)")
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(blue, cmap="Blues", vmin=0, vmax=v_max_bandes)
        axes[0, 2].set_title("Banda Blue (B02)")
        plt.colorbar(im2, ax=axes[0, 2])

        # Fila 2: NIR, Composició RGB i NDWI
        im3 = axes[1, 0].imshow(nir, cmap="magma", vmin=0, vmax=4000)
        axes[1, 0].set_title("Banda NIR (B08)")
        plt.colorbar(im3, ax=axes[1, 0])

        axes[1, 1].imshow(rgb)
        axes[1, 1].set_title("Composició Color Real (RGB)")

        im5 = axes[1, 2].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
        axes[1, 2].set_title("Resultado NDWI")
        plt.colorbar(im5, ax=axes[1, 2])

        plt.tight_layout()

        # Guardar la figura
        output_png = photos_dir / f"Anàlisi_Completa_{date_str}_{scene_id}.png"
        plt.savefig(output_png, dpi=150)
        plt.close(fig)

        print(f"  -> Resultat guardat a: {output_png.name}")

    except Exception as e:
        print(f"  ERROR en processar l'escena {scene_id}: {e}")

print("\nProcés d'extracció i anàlisi completat amb èxit!")