# Proyect
#
"""
===============
Imagen de hueso cortical
===============
En el siguiente script se muestra cómo cargar una imagen de hueso cortical,
aplicar una corrección gamma para mejorar el contraste, convertir la imagen
a escala de grises, detectar contornos y medir la longitud de una barra de escala.

### Resultado:
El lienzo de 1x4 mostrará:
1. La imagen original.
2. La imagen corregida con gamma.
3. La imagen en escala de grises.
4. La imagen binaria con las zonas claras resaltadas y los contornos trazados.
"""

# Se importan todas las herramientas necesarias
import numpy as np
import matplotlib
matplotlib.use('TKAgg')  # Uso el backend TKAgg para evitar advertencias en Linux.
import matplotlib.pyplot as plt
from skimage.io import imread  # Para cargar imágenes
from skimage.color import rgb2gray  # Para convertir imágenes a escala de grises
from skimage.exposure import adjust_gamma  # Para corregir el contraste
from skimage import measure, filters  # Para encontrar contornos y aplicar filtros
from skimage.filters import threshold_otsu  # Para aplicar un umbral a la imagen

# Cargar la imagen del hueso cortical
image = imread('/home/ccely/Documentos/IFLYSIB/BiblioTK/Bone_images/bonehuman_cooper2016.png')

# Si la imagen tiene 4 canales (RGBA), eliminamos el canal alfa
if image.shape[-1] == 4:
    image = image[..., :3]  # Seleccionamos solo los canales RGB

# Se aplica la corrección gamma
gamma_corrected = adjust_gamma(image, 2)

# Convertimos la imagen corregida con gamma a escala de grises
if gamma_corrected.ndim == 3:
    gray_image = rgb2gray(gamma_corrected)  # Convertimos la imagen corregida a escala de grises
else:
    gray_image = gamma_corrected  # Si ya está en escala de grises, la usamos directamente

# Aplicar un umbral para segmentar la barra blanca
threshold = threshold_otsu(gray_image)  # Calcula un umbral óptimo
binary_image = gray_image > threshold  # Segmenta las áreas claras (blancas)

# Encontrar las coordenadas de los píxeles blancos
coords = np.column_stack(np.where(binary_image))  # Coordenadas de los píxeles blancos

# Calcular la longitud de la barra blanca
x_min, x_max = coords[:, 1].min(), coords[:, 1].max()  # Extremos en el eje x
barra_length_pixels = x_max - x_min  # Longitud en píxeles
print(f"Longitud de la barra blanca: {barra_length_pixels} píxeles")

# Relación píxeles/µm basada en la barra de escala
scale_bar_length_um = 250  # Longitud conocida de la barra en micrómetros
pixels_per_um = barra_length_pixels / scale_bar_length_um  # Relación píxeles/µm
print(f"Relación píxeles/µm: {pixels_per_um:.2f}")

# Dimensiones reales de la imagen en micrómetros
image_width_pixels = image.shape[1]  # Ancho de la imagen en píxeles
image_height_pixels = image.shape[0]  # Alto de la imagen en píxeles
width_um = image_width_pixels / pixels_per_um
height_um = image_height_pixels / pixels_per_um
print(f"Dimensiones reales de la imagen: {width_um:.2f} µm x {height_um:.2f} µm")

# Aplicamos un umbral threshold para resaltar las zonas claras
threshold = 0.3  # Define el umbral
highlighted_image = gray_image > threshold  # Crea una imagen binaria donde las zonas claras son True

# Ahora se buscan los contornos en la imagen binaria (highlighted_image)
contours = measure.find_contours(highlighted_image, level=0.5)  # Nivel de intensidad para los contornos

# Aplicar un filtro Sobel a la imagen corregida con gamma
sobel_edges = filters.sobel(gray_image)  # Detectamos bordes en la imagen corregida

'''Para visualizar las imágenes se crea un lienzo de 1x4'''
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# ImgA. Imagen Original
axes[0].imshow(image)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

# ImgB. Imagen Corregida
axes[1].imshow(gamma_corrected)
axes[1].set_title('Imagen Corregida Gamma')
axes[1].axis('off')

# ImgC. Resultado del filtro Sobel
axes[2].imshow(sobel_edges, cmap='gray')
axes[2].set_title('Filtro Sobel')
axes[2].axis('off')

# ImgD. Imagen con las zonas claras resaltadas y contornos
axes[3].imshow(highlighted_image, cmap='gray')  # Mostrar la imagen binaria
for contour in contours:
    axes[3].plot(contour[:, 1], contour[:, 0], linewidth=1, color='purple')  # Trazar contornos en violeta
axes[3].set_title('Contornos en Zonas Claras')
axes[3].axis('off')

plt.tight_layout()  # Ajusta el espacio entre las imágenes
plt.show()

# Guardar las coordenadas de los píxeles blancos en un archivo de texto
np.savetxt('/home/ccely/Documentos/IFLYSIB/BiblioTK/Bone_images/coords.txt', coords, fmt='%d')  # Guardar coordenadas
