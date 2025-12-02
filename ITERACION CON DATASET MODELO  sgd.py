import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD

# ----------------------------------------------------
# 1. CARGAR DATASET
# ----------------------------------------------------
np.random.seed(2)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot
nclases = 10
y_train_cat = to_categorical(y_train, nclases)
y_test_cat = to_categorical(y_test, nclases)

# Redimensionar
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ----------------------------------------------------
# 2. DEFINIR FUNCIÓN PARA CREAR MODELO
# ----------------------------------------------------
def crear_modelo():
    modelo = Sequential()
    modelo.add(Input(shape=(28, 28, 1)))
    modelo.add(Conv2D(6, (5, 5), activation='relu'))
    modelo.add(MaxPooling2D((2, 2)))

    modelo.add(Conv2D(16, (5, 5), activation='relu'))
    modelo.add(MaxPooling2D((2, 2)))

    modelo.add(Flatten())
    modelo.add(Dense(120, activation='relu'))
    modelo.add(Dense(84, activation='relu'))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=0.1),
        metrics=['accuracy']
    )
    return modelo

# ----------------------------------------------------
# 3. EXPERIMENTO: PROBAR DIFERENTES TAMAÑOS DE DATASET
# ----------------------------------------------------
def probar_tamano_dataset(n):
    print(f"\n===== ENTRENANDO CON SOLO {n} IMÁGENES =====")
    idx = np.random.choice(len(x_train), n, replace=False)
    x_small = x_train[idx]
    y_small = y_train_cat[idx]

    modelo = crear_modelo()
    modelo.fit(x_small, y_small, epochs=5, batch_size=32, verbose=0)

    loss, acc = modelo.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Accuracy con {n} imágenes = {acc*100:.2f}%")

    return modelo, acc

# Tamaños a probar
tamaños = [10, 50, 100, 300, 500, 1000, 5000]

modelos_guardados = {}
accuracies = []

for n in tamaños:
    modelo, acc = probar_tamano_dataset(n)
    modelos_guardados[n] = modelo
    accuracies.append(acc)

# ----------------------------------------------------
# 4. CARGAR TU IMAGEN Y PROCESARLA
# ----------------------------------------------------

# *** AQUI PONES LA RUTA DE TU IMAGEN ***
RUTA_IMAGEN = r"D:..\\PROYECTO CARIOTIPO\imagen numero 6.png"

img = cv2.imread(RUTA_IMAGEN, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray")
plt.title("Imagen original")
plt.axis("off")
plt.show()

# procesar imagen igual que MNIST
img_proc = cv2.bitwise_not(img)
img_proc = cv2.resize(img_proc, (28, 28))
img_proc = img_proc / 255.0
img_proc = img_proc.reshape(1, 28, 28, 1)

print("\n====== RESULTADOS POR TAMAÑO DE DATASET ======\n")

# ----------------------------------------------------
# 5. PREDICCIONES CON CADA MODELO
# ----------------------------------------------------
for size, model in modelos_guardados.items():

    preds = model.predict(img_proc)

    print(f"Con {size} imágenes →")
    print("Predicciones (probabilidades):")
    print(preds)
    print("Número predicho:", np.argmax(preds))
    print("-" * 60)
