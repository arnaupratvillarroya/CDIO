from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# --- 1. Definir el directori de treball ---
photos_dir = Path("/home/cbltic/Escriptori/CDIO_1/CDIO/Photos")

# --- 2. Cercar tots els fitxers de la banda "green" ---
# Cerca qualsevol fitxer que acabi en '_green.tif'
green_files = sorted(list(photos_dir.glob("*_green.tif")))

if not green_files:
    print(f"No s'han trobat imatges *_green.tif a la carpeta: {photos_dir}")
else:
    print(f"S'han trobat {len(green_files)} parells d'imatges per processar.\n")

# --- 3. Iterar sobre cadascuna de les N imatges ---
for i, green_path in enumerate(green_files, 1):
    # Generar el nom del fitxer NIR corresponent substituint '_green.tif' per '_nir.tif'
    nir_path = Path(str(green_path).replace("_green.tif", "_nir.tif"))

    # Comprovar si existeix el parell NIR
    if not nir_path.exists():
        print(f"[{i}/{len(green_files)}] AVÍS: No s'ha trobat el parell NIR per a {green_path.name}. S'omet.")
        continue

    print(f"[{i}/{len(green_files)}] Processant: {green_path.stem}...")

    # --- 4. Carregar les bandes ---
    with rasterio.open(green_path) as src_green:
        green = src_green.read(1)

    with rasterio.open(nir_path) as src_nir:
        nir = src_nir.read(1)

    # --- 5. Calcular el NDWI ---
    green_f = green.astype(float)
    nir_f = nir.astype(float)

    denominador = green_f + nir_f
    denominador[denominador == 0] = 1e-8

    ndwi = (green_f - nir_f) / denominador

    # --- 6. Visualitzar i guardar el resultat ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Definir un títol general amb la data/nom de la imatge
    base_name = green_path.name.replace("_green.tif", "")
    fig.suptitle(f"Anàlisi NDWI: {base_name}", fontsize=14)

    # Banda Verde
    im0 = axes[0].imshow(green, cmap="Greens")
    axes[0].set_title("Banda Verde (Green)")
    plt.colorbar(im0, ax=axes[0])

    # Banda NIR
    im1 = axes[1].imshow(nir, cmap="magma")
    axes[1].set_title("Banda NIR")
    plt.colorbar(im1, ax=axes[1])

    # NDWI
    im2 = axes[2].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
    axes[2].set_title("Resultado NDWI")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    # Guardar el gràfic per a cada imatge amb el seu nom corresponent
    output_png = photos_dir / f"ndwi_{base_name}.png"
    plt.savefig(output_png)

    # Tanquem la figura per alliberar memòria RAM en processar moltes imatges
    plt.close(fig)

    print(f"  -> Resultat guardat a: {output_png.name}")

print("\nProcés finalitzat amb èxit!")