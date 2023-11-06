import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los datos desde dataset.csv
data = pd.read_csv('dataset02.csv')

# Separar los datos en X (características) y y (etiquetas)
X = data.drop(columns=['Unnamed: 0'])
y = pd.read_csv('y2.txt')

# Reducción de dimensionalidad con PCA para diferentes valores de d
d_values = [5, 10, 15, 20, X.shape[1]]
similarities = []

for d in d_values:
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    distance_matrix = euclidean_distances(X_reduced, X_reduced)  # Cambio aquí
    similarities.append(distance_matrix)

# Visualizar las matrices de similitud
fig, axes = plt.subplots(1, len(d_values), figsize=(15, 4))

for i, d in enumerate(d_values):
    im = axes[i].imshow(similarities[i], cmap='viridis')
    axes[i].set_title(f'd={d}')
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

# Identificar las dimensiones originales más representativas
n_important_dimensions = 10  # Puedes ajustar esto
important_dimensions = np.argsort(pca.components_)[:, -n_important_dimensions:]

# Imprimir las dimensiones originales más representativas
print("Dimensiones originales más representativas:")
for dimension in important_dimensions:
    print(f"Dimensión {dimension}:", data.columns[dimension])

model = LinearRegression()
model.fit(X_reduced, y)
y_pred = model.predict(X_reduced)
# Calcular el Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calcular el coeficiente de determinación R^2
r2 = r2_score(y, y_pred)
print("Coeficiente de Determinación (R^2):", r2)