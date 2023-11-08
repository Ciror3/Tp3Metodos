import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
# from google.colab import drive

# # Montar el drive y cargar los datos
# drive.mount('/content/drive')
# folder_path = '/content/drive/My Drive/Colab Notebooks/'
# file_name_X = 'dataset02.csv'
# file_name_Y = 'y2.txt'
# file_path_X = folder_path + file_name_X
# file_path_Y = folder_path + file_name_Y

data_X = pd.read_csv("dataset02.csv", header=0, index_col=0)
data_Y = np.loadtxt('y2.txt')

# Estandarizar los datos
scaler = StandardScaler()
data_X_std = scaler.fit_transform(data_X)

# Análisis PCA
d_pca = data_X_std.shape[1]
pca = PCA(n_components=d_pca)
pca.fit(data_X_std)
singular_values = pca.singular_values_
relative_importance = singular_values / np.sum(singular_values) * 100
cumulative_importance = np.cumsum(relative_importance)

# Gráfico de Importancia Acumulativa
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cumulative_importance) + 1), cumulative_importance)
plt.title(f'Importancia Acumulativa para d={d_pca}')
plt.xlabel('Número de Dimensiones')
plt.ylabel('Importancia Acumulativa (%)')
d_optima = np.argmax(cumulative_importance >= 80) + 1
plt.axvline(x=d_optima, color='red', linestyle='--', label=f'd óptima = {d_optima}')
plt.axhline(y=80, color='green', linestyle='--', label='80% de Importancia Acumulativa')
plt.legend()
plt.show()

# Matriz de Similaridad Alta Dimensión
similarity_high_dim = euclidean_distances(data_X_std)
plt.figure(figsize=(8, 6))
plt.imshow(similarity_high_dim, cmap='hot')
plt.title('Matriz de Similaridad (Alta Dimensión)')
plt.colorbar(label='Value')
plt.show()

# Reducción de Dimensionalidad con PCA (d = 2)
pca = PCA(n_components=2)
data_reduced_pca = pca.fit_transform(data_X_std)
plt.scatter(data_X_std[:, 0], data_X_std[:, 1], c='orange', label='Datos Originales')
plt.scatter(data_reduced_pca[:, 0], data_reduced_pca[:, 1], c='blue', label='PCA (d=2)')
plt.title('Datos Originales vs PCA (d = 2)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(loc='best')
plt.show()

# Reducción de Dimensionalidad con PCA (d = 2) directo desde data_std
pca = PCA(n_components=2)
data_std_reduced = pca.fit_transform(data_X_std)
plt.scatter(data_reduced_pca[:, 0], data_reduced_pca[:, 1], c='blue', label='PCA (d=2) de PCA (d=106)')
plt.scatter(data_std_reduced[:, 0], data_std_reduced[:, 1], c='orange', label='PCA (d=2) de data_std')
plt.title('Datos Reducidos desde PCA (d=106) a PCA (d=2) vs Datos Reducidos directamente desde data_std a PCA (d=2)')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Matrices de Similaridad en Dimensiones Reducidas con PCA
dimensions = [78]
similarities_reduced_pca = []

for d in dimensions:
    pca = PCA(n_components=d)
    data_reduced_pca = pca.fit_transform(data_X_std)
    similarity_pca = euclidean_distances(data_reduced_pca)
    similarities_reduced_pca.append(similarity_pca)

plt.figure(figsize=(8, 6))
for i, d in enumerate(dimensions):
    plt.imshow(similarities_reduced_pca[i], cmap='hot')
    plt.title(f'PCA - d={d}')
    plt.colorbar()
plt.show()

# Regresión Lineal con PCA
dimensions = [78]
errors = []

for d in dimensions:
    VT_d = VT[:d]
    X_reduced = np.dot(data_X_std, VT_d.T)
    model = LinearRegression()
    model.fit(X_reduced, data_Y)
    Y_hat = model.predict(X_reduced)
    error = np.linalg.norm(Y_hat - data_Y)
    errors.append(error)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

x = range(len(data_Y))
x2 = range(len(Y_hat))
axes[0].scatter(x, data_Y, c="orange", label="Y")
axes[0].scatter(x2, Y_hat, c="blue", label="Y_hat")
axes[0].set_title('Y vs Y_hat para d = p')
axes[0].set_xlabel('Índice')
axes[0].set_ylabel('Valor')
axes[0].legend()

axes[1].plot(dimensions, errors, marker='o', linestyle='-')
axes[1].set_title('Error de Predicción vs. Dimensión Reducida (d)')
axes[1].set_xlabel('Dimensión Reducida (d)')
axes[1].set_ylabel('Error de Predicción')
axes[1].grid()

plt.tight_layout()
plt.show()
