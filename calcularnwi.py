import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 1. Rutas a las imágenes GeoTIFF (Ajusta las rutas según tus archivos)
ruta_verde = "S2C_31TDF_20250630_0_L2A_green.tif"  # o .tif (Banda Verde)
ruta_nir = "S2C_31TDF_20250630_0_L2A_nir.tif"    # o .tif (Banda Infrarrojo Cercano - NIR)

# 2. Cargar las bandas
with rasterio.open(ruta_verde) as src_green:
    banda_verde = src_green.read(1).astype(float)

with rasterio.open(ruta_nir) as src_nir:
    banda_nir = src_nir.read(1).astype(float)

# 3. Calcular el NDWI: (Green - NIR) / (Green + NIR)
# Se añade 1e-10 en el denominador para evitar la división por cero
denominador = banda_verde + banda_nir
denominador[denominador == 0] = 1e-10

ndwi = (banda_verde - banda_nir) / denominador

# 4. Calcular percentiles para ajustar el contraste visual de las bandas individuales
p2_verde, p98_verde = np.percentile(banda_verde, (2, 98))
p2_nir, p98_nir = np.percentile(banda_nir, (2, 98))

# 5. Representación gráfica de los resultados
plt.figure(figsize=(15, 5))

# Subplot 1: Banda Verde con contraste mejorado
plt.subplot(1, 3, 1)
plt.imshow(banda_verde, cmap='gray', vmin=p2_verde, vmax=p98_verde)
plt.title("Banda Verde (Green)")
plt.axis('off')

# Subplot 2: Banda Infrarrojo (NIR) con contraste mejorado
plt.subplot(1, 3, 2)
plt.imshow(banda_nir, cmap='gray', vmin=p2_nir, vmax=p98_nir)
plt.title("Banda Infrarrojo (NIR)")
plt.axis('off')

# Subplot 3: Mapa de NDWI
plt.subplot(1, 3, 3)
im = plt.imshow(ndwi, cmap='Blues', vmin=-1.0, vmax=1.0)
plt.title("NDWI (Índice de Agua)")
plt.axis('off')

# Barra de color para el NDWI
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Valor NDWI')

plt.tight_layout()
plt.show()