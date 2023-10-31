import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
imagen = Image.open("img00.jpeg")

# Obtiene las dimensiones de la imagen (ancho y alto)
p, alto = imagen.size

# Imprime el tamaño en formato (ancho, alto)
print(f"El tamaño de la imagen es ({p}, {alto})")

# Cargar y2.txt (asegúrate de adaptar esto según el formato de tus datos)
data = np.loadtxt('y2.txt', delimiter=' ')
U, S, VT = np.linalg.svd(data, full_matrices=False)
d = 50  # Define el número de dimensiones deseadas

compressed_U = U[:, :d]
compressed_S = np.diag(S[:d])
compressed_VT = VT[:d, :]

reconstructed_data = np.dot(compressed_U, np.dot(compressed_S, compressed_VT))

# Redimensionar los datos reconstruidos en imágenes
reconstructed_images = reconstructed_data.reshape(-1, p, p)

# Visualizar algunas imágenes originales y reconstruidas
num_images_to_display = 5
for i in range(num_images_to_display):
    plt.subplot(2, num_images_to_display, i + 1)
    plt.imshow(data[i].reshape(p, p), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, num_images_to_display, i + num_images_to_display + 1)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.title(f'Reconstructed (d={d})')
    plt.axis('off')

plt.show()
