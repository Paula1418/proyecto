import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, color, exposure, img_as_ubyte

# --- Carpeta de entrada ---
carpeta_entrada = r"D:\Documentos\Programacion\PROYECTO CARIOTIPO\CariotipoDataset\segmentacion"

# --- Carpeta de salida ---
carpeta_salida = "cromosomas_recortados"
os.makedirs(carpeta_salida, exist_ok=True)

extensiones = (".png", ".jpg", ".jpeg", ".tif", ".bmp")


# ---------------------------------------------------------
# FUNCIÓN PARA PROCESAR UNA IMAGEN
# ---------------------------------------------------------
def procesar_imagen(ruta_imagen):
    print(f"\nProcesando: {ruta_imagen}")

    # --- Cargar y normalizar ---
    image = io.imread(ruta_imagen, as_gray=True)
    image_norm = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
    image_eq = exposure.equalize_adapthist(image_norm, clip_limit=0.08, kernel_size=(32, 32))

    # --- Binarización y limpieza ---
    thresh = filters.threshold_otsu(image_eq)
    binary = image_eq < (thresh * 1)

    binary = morphology.remove_small_objects(binary, min_size=90)
    binary = morphology.remove_small_holes(binary, area_threshold=200)
    binary = morphology.binary_dilation(binary, morphology.disk(0.95))

    # --- Fondo blanco ---
    binary_inv = np.invert(binary)
    image_fondo_blanco = image_norm.copy()
    image_fondo_blanco[binary == 0] = 1.0

    # --- Etiquetado ---
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)

    # --- Filtrar cromosomas ---
    areas = [p.area for p in props]
    area_promedio = np.mean(areas)

    labeled_filtrado = np.zeros_like(labeled)
    contador = 0

    for prop in props:
        if prop.area > 0.04 * area_promedio:   # LÍNEA AJUSTABLE
            contador += 1
            labeled_filtrado[labeled == prop.label] = contador

    print(f"   → Se detectaron {contador} cromosomas")

    # --- Carpeta individual ---
    nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
    carpeta_individual = os.path.join(carpeta_salida, nombre_base)
    os.makedirs(carpeta_individual, exist_ok=True)

    # --- Guardar recortes ---
    for prop in measure.regionprops(labeled_filtrado):
        minr, minc, maxr, maxc = prop.bbox
        cromosoma = image_fondo_blanco[minr:maxr, minc:maxc]
        nombre = f"{carpeta_individual}/cromosoma_{prop.label}.png"
        io.imsave(nombre, img_as_ubyte(cromosoma))

    # --- Mostrar segmentación ---
    labeled_rgb = color.label2rgb(labeled_filtrado, bg_label=0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(image_norm, cmap='gray')
    axes[0].set_title("Imagen normalizada")
    axes[0].axis('off')

    axes[1].imshow(binary_inv, cmap='gray')
    axes[1].set_title("Imagen binaria (fondo blanco)")
    axes[1].axis('off')

    axes[2].imshow(image_fondo_blanco, cmap='gray')
    axes[2].set_title("Cromosomas sobre fondo blanco")
    axes[2].axis('off')

    axes[3].imshow(labeled_rgb)
    axes[3].set_title(f"Segmentación y numeración ({contador})")
    axes[3].axis('off')

    # Numerar cromosomas
    for prop in measure.regionprops(labeled_filtrado):
        y, x = prop.centroid
        axes[3].text(x, y, str(prop.label), color='white', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # --- Guardar resultados ---
    io.imsave(f"{carpeta_individual}/imagen_binaria.png", img_as_ubyte(binary_inv))
    io.imsave(f"{carpeta_individual}/cariotipo_fondo_blanco.png", img_as_ubyte(image_fondo_blanco))
    io.imsave(f"{carpeta_individual}/cariotipo_segmentado.png", img_as_ubyte(labeled_rgb))

    print(f"   → Resultados guardados en: {carpeta_individual}")


# ---------------------------------------------------------
# PROCESAR TODAS LAS IMÁGENES DE LA CARPETA
# ---------------------------------------------------------
imagenes = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(extensiones)]

print(f"\nTotal de imágenes encontradas: {len(imagenes)}")

for img in imagenes:
    ruta = os.path.join(carpeta_entrada, img)
    procesar_imagen(ruta)

print("\nTODAS LAS IMÁGENES HAN SIDO PROCESADAS.")
