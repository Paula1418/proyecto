import cv2
import numpy as np
from skimage import measure, morphology
import pandas as pd
import joblib

# -------------------------------------------------------
# Cargar modelo y scaler para 41/42
# -------------------------------------------------------
modelo = joblib.load("modelo_cromosomas_41_42.pkl")
scaler = joblib.load("scaler_41_42.pkl")

# -------------------------------------------------------
# Segmentación
# -------------------------------------------------------
def segmentar_cromosomas(ruta):
    img_color = cv2.imread(ruta)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {ruta}")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thr_clean = morphology.remove_small_objects(thr.astype(bool), min_size=100)
    thr_clean = thr_clean.astype(np.uint8) * 255

    labels = measure.label(thr_clean, connectivity=2)
    props = measure.regionprops(labels)

    cromosomas = []
    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        w = maxc - minc
        h = maxr - minr
        pixels = prop.area
        cromosomas.append({
            "ID": i,
            "Ancho": w,
            "Alto": h,
            "Pixeles": pixels,
            "bbox": (minr, minc, maxr, maxc)
        })
    return pd.DataFrame(cromosomas), img_color

# -------------------------------------------------------
# Predicción
# -------------------------------------------------------
def predecir_41_42(df):
    X = df[["Alto", "Ancho", "Pixeles"]]
    X_scaled = scaler.transform(X)
    df["Pred"] = modelo.predict(X_scaled)
    df["Prob_41_42"] = modelo.predict_proba(X_scaled)[:, 1]
    return df

# -------------------------------------------------------
# Función para redimensionar la imagen y que quepa en pantalla
# -------------------------------------------------------
def resize_to_fit_screen(image, max_width=1400, max_height=800):
    """
    Redimensiona la imagen manteniendo la proporción para que no exceda
    max_width x max_height. Si ya es más pequeña, no la agranda.
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    ruta = "D:/Documentos/Programacion/PROYECTO CARIOTIPO/CariotipoDataset/segmentacion"

    df, img_vis = segmentar_cromosomas(ruta)
    df = predecir_41_42(df)

    # Mostrar solo los más probables (top 2)
    top_indices = df.nlargest(2, "Prob_41_42").index

    for idx, fila in df.iterrows():
        minr, minc, maxr, maxc = fila["bbox"]
        prob = fila["Prob_41_42"]

        # Color: verde si es top 2, rojo si no
        if idx in top_indices:
            color = (0, 255, 0)  # Verde
            thickness = 2
        else:
            color = (0, 0, 255)  # Rojo
            thickness = 1

        # Dibujar rectángulo
        cv2.rectangle(img_vis, (minc, minr), (maxc, maxr), color, thickness)

        # Solo mostrar texto si es top 2 (para evitar desorden)
        if idx in top_indices:
            text = f"{fila['ID']}: {prob:.2f}"
            font_scale = 0.45
            thickness_text = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness_text)
            x, y = minc, minr - 5

            # Fondo blanco detrás del texto
            cv2.rectangle(img_vis, (x, y - h - 2), (x + w, y + 2), (255, 255, 255), -1)
            # Texto negro
            cv2.putText(img_vis, text, (x, y), font, font_scale, (0, 0, 0), thickness_text, cv2.LINE_AA)

    # 🔥 REDIMENSIONAR LA IMAGEN ANTES DE MOSTRARLA 🔥
    img_resized = resize_to_fit_screen(img_vis, max_width=1400, max_height=800)

    # Mostrar resultado ajustado
    cv2.imshow("Cromosomas 41/42 (X/Y) - Solo top 2 etiquetados", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()