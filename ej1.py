import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances

def load_images(image_directory):
    image_files = os.listdir(image_directory)
    image_list = []

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = Image.open(image_path)
        image = image.convert('L')
        image_array = np.array(image)
        image_list.append(image_array)

    p,_ = image_list[0].shape  # Obtener el tamaño de las imágenes
    return image_list, p

def perform_svd_and_reconstruction(image_list, p, d1, d2):
    n = len(image_list)
    data_matrix = np.zeros((n, p * p))

    for i in range(n):
        image = image_list[i].reshape(p * p)
        data_matrix[i, :] = image

    U, S, VT = svd(data_matrix, full_matrices=False)

    reconstructed_images_d1 = np.dot(U[:, :d1], np.dot(np.diag(S[:d1]), VT[:d1, :]))
    reconstructed_images_d2 = np.dot(U[:, :d2], np.dot(np.diag(S[:d2]), VT[:d2, :]))

    return U, S, VT, reconstructed_images_d1, reconstructed_images_d2

def calculate_and_plot_similarity(A):
    # Realizar la descomposición SVD en una matriz A
    U, S, VT = svd(A, full_matrices=False)

    # Calcular la similitud entre los vectores singulares izquierdos y derechos utilizando la distancia coseno
    similarity = cosine_distances(U, VT)

    # Crear el mapa de calor de la similitud
    plt.figure(figsize=(8, 8))
    plt.imshow(similarity, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Mapa de Similitud entre Vectores Singulares Izquierdos y Derechos')
    plt.show()

def visualize_images(image_matrix_list, num_images_to_display, p_list):
    fig, axes = plt.subplots(nrows=len(image_matrix_list), ncols=num_images_to_display, figsize=(12, 6))

    for i, image_matrix in enumerate(image_matrix_list):
        p = p_list[i]  # Obtener el tamaño de las imágenes para este conjunto
        for j in range(num_images_to_display):
            axes[i, j].imshow(image_matrix[j].reshape(p, p), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'd={d_values[i]}')

    plt.tight_layout()
    plt.show()

def calculate_cosine_similarity_matrices(U, S, VT, d_values):
    similarities = []
    for d in d_values:
        U_d = U[:, :d]
        S_d = np.diag(S[:d])
        VT_d = VT[:d, :]
        reconstructed_images = np.dot(U_d, np.dot(S_d, VT_d))
        similarity_matrix = 1 - cosine_distances(reconstructed_images)
        similarities.append(similarity_matrix)
    return similarities

def calculate_euclidean_similarity_matrices(U, S, VT, d_values):
    similarities = []
    for d in d_values:
        U_d = U[:, :d]
        S_d = np.diag(S[:d])
        VT_d = VT[:d, :]
        reconstructed_images = np.dot(U_d, np.dot(S_d, VT_d))
        similarity_matrix = 1 / (1 + pairwise_distances(reconstructed_images, metric='euclidean'))
        similarities.append(similarity_matrix)
    return similarities

def create_similarity_heatmap(similarity_matrix):
    x_labels = [f'{i}' for i in range(similarity_matrix.shape[0])]
    y_labels = [f'{i}' for i in range(similarity_matrix.shape[1])]

    plt.figure(figsize=(8, 8))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='none')
    plt.xticks(np.arange(similarity_matrix.shape[0]), x_labels, rotation=90)
    plt.yticks(np.arange(similarity_matrix.shape[1]), y_labels)
    plt.colorbar()
    plt.show()

def find_minimal_d_with_error(data_matrix, max_error):
    U, S, VT = svd(data_matrix, full_matrices=False)
    d_values = []
    errors = []
    d = 1
    error = 1.0

    while error > max_error:
        U_d = U[:, :d]
        S_d = np.diag(S[:d])
        VT_d = VT[:d, :]
        reconstructed_image = np.dot(U_d, np.dot(S_d, VT_d))
        error = np.linalg.norm(data_matrix - reconstructed_image, 'fro') / np.linalg.norm(data_matrix, 'fro')
        d_values.append(d)
        errors.append(error)
        d += 1
    return d_values, errors

def plot_images(images, titles):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def apply_compression_to_all_images(image_list, max_error):
    compressed_images = []
    for image in image_list:
        p = image.shape[0]
        image_vector = image.reshape(-1)
        data_matrix = image_vector.reshape(1, -1)
        d_values, _ = find_minimal_d_with_error(data_matrix, max_error)
        d = d_values[-1]  # Get the last d value that meets the error criteria
        U, S, VT = svd(image_vector.reshape(1, -1), full_matrices=False)
        U_d = U[:, :d]
        S_d = np.diag(S[:d])
        VT_d = VT[:d, :]
        reconstructed_image = np.dot(U_d, np.dot(S_d, VT_d))
        compressed_images.append(reconstructed_image.reshape(image.shape))

    return compressed_images

def transform_svd_matrix(image_list, p, k):
    n = len(image_list)
    data_matrix = np.zeros((n, p * p))

    for i in range(n):
        image = image_list[i].reshape(p * p)
        data_matrix[i, :] = image

    U, S, VT = svd(data_matrix, full_matrices=False)

    reconstructed_images_upper = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))
    reconstructed_images_lower = np.dot(U[:, -k:], np.dot(np.diag(S[-k:]), VT[-k:, :]))

    return reconstructed_images_upper, reconstructed_images_lower

def show_original_and_reconstructed_images(image_list, p, k, num_images_to_display):
    reconstructed_images_upper, reconstructed_images_lower = transform_svd_matrix(image_list, p, k)

    fig, axes = plt.subplots(num_images_to_display, 3, figsize=(15, 5))

    for i in range(num_images_to_display):
        # Imagen original
        axes[i, 0].imshow(image_list[i].reshape(p, p), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Imagen reconstruida utilizando los primeros k componentes singulares
        axes[i, 1].imshow(reconstructed_images_upper[i].reshape(p, p), cmap='gray')
        axes[i, 1].set_title(f'k={k} primeros')
        axes[i, 1].axis('off')

        # Imagen reconstruida utilizando los últimos k componentes singulares
        axes[i, 2].imshow(reconstructed_images_lower[i].reshape(p, p), cmap='gray')
        axes[i, 2].set_title(f'k={k} ultimos')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_directory = 'dataset_imagenes'
    max_error = 0.1
    d1, d2 = 10, 20

    image_list, p = load_images(image_directory)
    U, S, VT, reconstructed_images_d1, reconstructed_images_d2 = perform_svd_and_reconstruction(image_list, p, d1, d2)
    d_values = [5, 10, 15, 20, 25]
    num_images_to_display = 15  
    p_values = [p, p, p] 
    visualize_images([image_list, reconstructed_images_d1, reconstructed_images_d2], num_images_to_display, p_values)
    similarities = calculate_cosine_similarity_matrices(U, S, VT, d_values)
    euclidean_similarities = calculate_euclidean_similarity_matrices(U, S, VT, d_values)
    create_similarity_heatmap(euclidean_similarities[-1])

    data_matrix = np.zeros((len(image_list), image_list[0].shape[0] * image_list[0].shape[1]))
    for i, image in enumerate(image_list):
        data_matrix[i, :] = image.reshape(-1)

    d_values, errors = find_minimal_d_with_error(data_matrix, max_error)

    plt.figure(figsize=(8, 6))
    plt.plot(d_values, errors, marker='o')
    plt.xlabel('Número de Dimensiones (d)')
    plt.ylabel('Error (Norma de Frobenius)')
    plt.title('Error de Compresión vs. Número de Dimensiones')
    plt.grid(True)
    plt.show()

    compressed_images = apply_compression_to_all_images(image_list, max_error)
    titles = [f'{i+1}' for i in range(len(compressed_images))]
    plot_images(compressed_images, titles)

    k = 4  # Cambia este valor según tus necesidades
    amount_images_to_display = 5
    # Supongamos que image_list y p ya están definidos
    show_original_and_reconstructed_images(image_list, p, k, amount_images_to_display)



