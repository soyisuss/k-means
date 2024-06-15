import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Cargar el archivo de datos
dataframe = pd.read_csv('C:/Users/cbchz/OneDrive/Documentos/School/Code/IA/Practica10/global_index.csv')

# Obtiene las características de interés
X = dataframe[['Criminality', 'Criminal markets', 'Criminal actors']]

# El Codo
modelo_prueba = KMeans(random_state=42, n_init=10)
elbow_visualizer = KElbowVisualizer(modelo_prueba, k=(1, 11))
elbow_visualizer.fit(X)
elbow_visualizer.show()
best_k = elbow_visualizer.elbow_value_
print(f"El mejor valor para k de acuerdo con elbow es: {best_k}")

# Silueta
for k in range(2, 11):
    modelo_prueba = KMeans(n_clusters=k, random_state=42, n_init=10)
    silhouette_visualizer = SilhouetteVisualizer(modelo_prueba, colors='yellowbrick')
    silhouette_visualizer.fit(X)
    score = silhouette_visualizer.silhouette_score_
    silhouette_visualizer.show()
    print(f"Silhouette Score para k={k}: {score:.2f}")

plt.show()
