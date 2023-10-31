import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_distances
from numpy.linalg import norm

# Directorio que contiene las imágenes descomprimidas
image_directory = 'dataset_imagenes'

# Obtener la lista de nombres de archivos de imágenes en el directorio
image_files = os.listdir(image_directory)

# Lista para almacenar las imágenes
image_list = []

# Cargar las imágenes y convertirlas en arreglos NumPy
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = Image.open(image_path)  # Abrir la imagen con Pillow
    image = image.convert('L')  # Convertir a escala de grises
    image_array = np.array(image)  # Convertir a un arreglo NumPy
    image_list.append(image_array)

# Crear una matriz de datos para almacenar las imágenes
n = len(image_list)  # Número de imágenes
p = image_array.shape[0]  # Tamaño de las imágenes (p × p)

data_matrix = np.zeros((n, p * p))

# Convertir y apilar las imágenes en la matriz de datos
for i in range(n):
    image = image_list[i].reshape(p * p)
    data_matrix[i, :] = image

# Realizar la descomposición SVD
U, S, VT = svd(data_matrix, full_matrices=False)

# Definir un valor de d para la compresión
d1 = 10  # Por ejemplo, utiliza las primeras 10 dimensiones
d2 = 20  # Por ejemplo, utiliza las primeras 100 dimensiones

# Reconstruir imágenes con las primeras y últimas dimensiones
reconstructed_images_d1 = np.dot(U[:, :d1], np.dot(np.diag(S[:d1]), VT[:d1, :]))
reconstructed_images_d2 = np.dot(U[:, :d2], np.dot(np.diag(S[:d2]), VT[:d2, :]))

# Visualizar imágenes originales y reconstruidas
num_images_to_display = 15  # Número de imágenes a mostrar
fig, axes = plt.subplots(nrows=3, ncols=num_images_to_display, figsize=(12, 6))

for i in range(num_images_to_display):
    axes[0, i].imshow(data_matrix[i].reshape(p, p), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(reconstructed_images_d1[i].reshape(p, p), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f'd={d1}')

    axes[2, i].imshow(reconstructed_images_d2[i].reshape(p, p), cmap='gray')
    axes[2, i].axis('off')
    axes[2, i].set_title(f'd={d2}')

plt.tight_layout()
plt.show()

# Crear una lista de valores d para la compresión
d_values = [10, 20, 30, 40, 50]  # Puedes ajustar esto

# Lista para almacenar las matrices de similaridad
similarities = []

for d in d_values:
    # Realizar la descomposición SVD con d dimensiones
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = VT[:d, :]

    # Reconstruir imágenes con d dimensiones
    reconstructed_images = np.dot(U_d, np.dot(S_d, VT_d))

    # Calcular la matriz de similaridad utilizando la distancia coseno 
    similarity_matrix = 1 - cosine_distances(reconstructed_images)
    similarities.append(similarity_matrix)# Crear una lista de valores d para la compresión
d_values = [10, 20, 30, 40, 50]  # Puedes ajustar esto

# Lista para almacenar las matrices de similaridad
similarities = []

for d in d_values:
    # Realizar la descomposición SVD con d dimensiones
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = VT[:d, :]

    # Reconstruir imágenes con d dimensiones
    reconstructed_images = np.dot(U_d, np.dot(S_d, VT_d))

    # Calcular la matriz de similaridad utilizando la distancia coseno
    similarity_matrix = 1 - cosine_distances(reconstructed_images)
    similarities.append(similarity_matrix)

# Crear el mapa de calor
labels = [str(i) for i in range(similarity_matrix.shape[0])]

plt.figure(figsize=(8, 8))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='none')

# Configurar etiquetas de los ejes (opcional)
plt.xticks(np.arange(similarity_matrix.shape[0]))
plt.yticks(np.arange(similarity_matrix.shape[1]))

# Configurar los valores de las etiquetas de los ejes (opcional)
plt.xticks(np.arange(similarity_matrix.shape[0]), 'x', rotation=90)
plt.yticks(np.arange(similarity_matrix.shape[1]), 'y')

# Mostrar una barra de color para la escala (opcional)
plt.colorbar()

# Mostrar el mapa de calor
plt.show()