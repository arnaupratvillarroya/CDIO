import matplotlib.pyplot as plt
import numpy as np

# Si utilizas rasterio para abrir imágenes satelitales GeoTIFF:
import rasterio

# --- 1. Cargar ambas bandas con sus nombres de archivo reales ---
with rasterio.open('S2C_31TDF_20250630_0_L2A_green.tif') as src_green:
    green = src_green.read(1)

with rasterio.open('S2C_31TDF_20250630_0_L2A_nir.tif') as src_nir:
    nir = src_nir.read(1)

# --- 2. Calcular el NDWI (Band Math) ---
# Convertimos a float para evitar problemas de divisiones con enteros
green_f = green.astype(float)
nir_f = nir.astype(float)

# Evitar división por cero
denominador = green_f + nir_f
denominador[denominador == 0] = 1e-8

ndwi = (green_f - nir_f) / denominador

# --- 3. Visualizar con Matplotlib ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Mostrar Banda Verde
im0 = axes[0].imshow(green, cmap="Greens")
axes[0].set_title("Banda Verde (Green)")
plt.colorbar(im0, ax=axes[0])

# Mostrar Banda NIR
im1 = axes[1].imshow(nir, cmap="magma")
axes[1].set_title("Banda NIR")
plt.colorbar(im1, ax=axes[1])

# Mostrar Resultado NDWI
im2 = axes[2].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
axes[2].set_title("Resultado NDWI")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("resultado_ndwi.png")
plt.show()