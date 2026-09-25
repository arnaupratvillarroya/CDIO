from pystac_client import Client
import geopandas as gpd

# Cargar AOI
aoi = gpd.read_file("../data/Castelldefels_map.geojson")
aoi_geometry = aoi.geometry.iloc[0].__geo_interface__

# Conectar con STAC
catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

# Buscar imágenes
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=aoi_geometry,
    datetime="2025-01-01/2025-07-01"
)

items = list(search.items())

print(f"Imágenes encontradas: {len(items)}")

# Mostrar las propiedades de la primera imagen
item = items[0]

print("\nID de la imagen:")
print(item.id)

print("\nPropiedades:")
for key, value in item.properties.items():
    print(f"{key}: {value}")