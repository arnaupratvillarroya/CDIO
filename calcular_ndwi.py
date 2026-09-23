import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cargar_banda(ruta_archivo):
    """Carga una banda GeoTIFF y la convierte a array de NumPy."""
    img = Image.open(ruta_archivo)
    return np.array(img, dtype=np.float32)

# Archivos de la imagen Sentinel-2
archivo_green = "S2C_31TDF_20250630_0_L2A_green.tif"
archivo_nir = "S2C_31TDF_20250630_0_L2A_nir.tif"

print("Cargando bandas...")
banda_green = cargar_banda(archivo_green)
banda_nir = cargar_banda(archivo_nir)

print("Calculando NDWI...")
denominador = banda_green + banda_nir
denominador[denominador == 0] = 1e-8  # Evitamos división por cero

# Fórmula del NDWI
ndwi = (banda_green - banda_nir) / denominador

print("Mostrando resultado...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Banda Verde (Green)")
plt.imshow(banda_green, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Banda Infrarrojo (NIR)")
plt.imshow(banda_nir, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("NDWI (Índice de Agua)")
im = plt.imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
plt.colorbar(im, label='Valor NDWI')
plt.axis('off')

plt.tight_layout()
plt.show()