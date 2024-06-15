import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Cargar datos geoespaciales
path = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(path)

# Cargar datos del archivo CSV
dataframe = pd.read_csv('C:/Users/cbchz/OneDrive/Documentos/School/Code/IA/Practica10/global_index.csv')

# Obtener los nombres de los países del archivo CSV
countries = dataframe['Country'].unique()

# Crear un mapa base
ax = world.plot()

# Colores para diferentes países
colors = ['lightgreen', 'green', 'violet', 'rebeccapurple']

# Mapear y colorear los países del CSV en el mapa
for country, color in zip(countries, colors):
    world[world.name == country].plot(color=color, ax=ax)

# Guarda el mapa en una imagen PNG
plt.savefig('crime_map.png', dpi=300)
plt.show()
