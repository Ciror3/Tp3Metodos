import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import zipfile

with zipfile.ZipFile('dataset_imagenes.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset_imagenes')

# Cargar imágenes en una matriz de datos
image_paths = ['dataset_imagenes/image1.jpg', 'dataset_imagenes/image2.jpg', ...]  # Lista de rutas de imágenes
n = len(image_paths)  # Número de imágenes
p = 64  # Tamaño de las imágenes (p × p)

# Crear una matriz de datos para almacenar las imágenes
data_matrix = np.zeros((n, p * p))

# Cargar y convertir las imágenes en vectores
for i, image_path in enumerate(image_paths):
    image = plt.imread(image_path)
    data_matrix[i] = image.ravel()

# Realizar la descomposición SVD
U, S, VT = svd(data_matrix, full_matrices=False)

# Número de dimensiones a utilizar (por ejemplo, 10 y 50)
d1 = 10
d2 = 50

# Reconstruir imágenes con las primeras y últimas dimensiones
reconstructed_images1 = np.dot(U[:, :d1], np.dot(np.diag(S[:d1]), VT[:d1, :]))
reconstructed_images2 = np.dot(U[:, :d2], np.dot(np.diag(S[:d2]), VT[:d2, :]))

# Visualizar imágenes reconstruidas
plt.figure(figsize=(12, 6))
for i in range(5):  # Mostrar las primeras 5 imágenes
    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(data_matrix[i].reshape(p, p), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(reconstructed_images1[i].reshape(p, p), cmap='gray')
    plt.title(f'Reconstruida (d={d1})')
    plt.axis('off')

    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(reconstructed_images2[i].reshape(p, p), cmap='gray')
    plt.title(f'Reconstruida (d={d2})')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Calcular la matriz de similaridad para diferentes valores de d
d_values = [10, 20, 30, 40, 50]  # Puedes ajustar esto
similarities = []

for d in d_values:
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = VT[:d, :]
    reconstructed_images_d = np.dot(U_d, np.dot(S_d, VT_d))

    # Calcular la matriz de similaridad (puedes usar otras métricas)
    similarity_matrix = 1 - pairwise_distances(reconstructed_images_d, metric='cosine')
    similarities.append(similarity_matrix)

# Visualizar las matrices de similaridad para diferentes valores de d (opcional)
for i, d in enumerate(d_values):
    plt.figure()
    plt.imshow(similarities[i], cmap='viridis')
    plt.title(f'Similaridad (d={d})')
    plt.colorbar()
    plt.show()

# Definir la imagen de referencia (por ejemplo, la primera imagen)
reference_image = data_matrix[0]

# Definir el umbral de error
error_threshold = 0.10  # 10%

# Inicializar d
d_min = 1

# Iterar para encontrar d mínimo
for d in range(1, min(data_matrix.shape) + 1):
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = VT[:d, :]
    reconstructed_image = np.dot(U_d, np.dot(S_d, VT_d))

    # Calcular el error entre la imagen comprimida y la original
    reconstruction_error = np.linalg.norm(reference_image - reconstructed_image, 'fro') / np.linalg.norm(reference_image, 'fro')

    if reconstruction_error <= error_threshold:
        d_min = d
    else:
        break

# Comprimir todas las imágenes con d_min
compressed_images = np.dot(U[:, :d_min], np.dot(np.diag(S[:d_min]), VT[:d_min, :]))
