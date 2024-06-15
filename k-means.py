import os
os.environ["OMP_NUM_THREADS"] = "1"
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from tkinter import simpledialog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Verifica el directorio de trabajo actual
print(f"Directorio de trabajo actual: {os.getcwd()}")

# Cargar el archivo de datos
dataframe = pd.read_csv('C:/Users/cbchz/OneDrive/Documentos/School/Code/IA/Practica10/global_index.csv')

# Obtiene las características de interés
X = dataframe[['Criminality', 'Criminal markets', 'Criminal actors']]

# Crear la ventana raíz
root = tk.Tk()
root.withdraw()

# Pide el valor de K
k = simpledialog.askinteger("Valor de K", "Introduce el valor de K:")

# Mostrar el número capturado
if k is not None:
    print(f"\nEl número capturado es: {k}")
else:
    print("\nNo se introdujo ningún número, se asignará k=4")
    k = 4

# Inicialización para algoritmo de clustering con el parámetro K indicado por el usuario
kmeansModel = KMeans(n_clusters=k, random_state=42, n_init=10)

# Carga los datos
kmeansModel.fit(X)

# Obtiene los centroides
centroides = kmeansModel.cluster_centers_

# Obtiene una lista con las etiquetas de los datos
etiqueta = kmeansModel.predict(X)

# Agrega al frame de datos una columna para las etiquetas de clasificación
dataframe['Class'] = etiqueta

# Crear una figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Colores para las diferentes clases
colors = ['magenta', 'lime', 'cyan', 'red', 'royalblue', 'chocolate', 'gold', 'darkviolet', 'forestgreen', 'gray']

# Graficar los datos
for i in range(k):
    # Seleccionar puntos de la clase i
    xs = dataframe[dataframe['Class'] == i]['Criminality']
    ys = dataframe[dataframe['Class'] == i]['Criminal markets']
    zs = dataframe[dataframe['Class'] == i]['Criminal actors']
    ax.scatter(xs, ys, zs, c=colors[i], label=f'Class {i}')

# Etiquetas de los ejes
ax.set_xlabel('Criminality')
ax.set_ylabel('Criminal markets')
ax.set_zlabel('Criminal actors')

# Título y leyenda
ax.set_title('Gráfica de dispersión')
ax.legend()
plt.show()

print("\nMétricas de rendimiento\n")

# Evaluar el coeficiente de silueta
silhouette_avg = silhouette_score(X, kmeansModel.labels_)
print(f"Coeficiente de silueta: {silhouette_avg:.4f} (Es mejor si está más cerca de 1)")

# Evaluar el índice de Davies-Bouldin
db_index = davies_bouldin_score(X, kmeansModel.labels_)
print(f"Índice de Davies-Bouldin: {db_index:.4f} (Es mejor si está más cerca de 0)")

# Evaluar el índice de Calinski-Harabasz
ch_index = calinski_harabasz_score(X, kmeansModel.labels_)
print(f"Índice de Calinski-Harabasz: {ch_index:.4f} (Entre más alto mejor)")

dataframe = dataframe.sort_values(by='Class', ascending=True)

dataframe.to_csv('global_index_clusters.csv', encoding='ISO-8859-1', index=False)

# Cargar datos geoespaciales
path = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(path)

# Unir los datos geoespaciales con los resultados del clustering
world = world.merge(dataframe, how='left', left_on='name', right_on='Country')

# Crear un mapa base
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Colorear los países de acuerdo a la clasificación
world.boundary.plot(ax=ax)
world.plot(column='Class', ax=ax, legend=True, categorical=True, cmap='viridis', edgecolor='black')

# Guarda el mapa en una imagen PNG
plt.savefig('crime_map.png', dpi=300)
plt.show()
