import os
import cv2
import numpy as np

print("Karla Saenz 0133 NL 52")

# Directorio base del script
dir_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_script)

nombres_imagen = ["loro_0133.jpg", "loro.jpg", "loro_0133.png", "loro.png"]

rutas_a_buscar = []
for nombre in nombres_imagen:
    rutas_a_buscar.append(os.path.join(dir_script, "imagenes", nombre))
    rutas_a_buscar.append(os.path.join(dir_script, "..", "imagenes", nombre))
    rutas_a_buscar.append(os.path.join(os.getcwd(), "imagenes", nombre))
    rutas_a_buscar.append(os.path.join(os.getcwd(), nombre))

imagen = None
for ruta in rutas_a_buscar:
    if os.path.exists(ruta):
        imagen = cv2.imread(ruta)
        if imagen is not None:
            break

if imagen is None:
    print("Error: No se encontro el archivo de la imagen del loro.")
    exit()

# 1. Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Detección de bordes mediante Canny
bordes = cv2.Canny(gris, 50, 150)

# 3. Configuración Base (Líneas Rojas)
lineas_base = cv2.HoughLinesP(
    bordes,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=10
)

resultado_base = imagen.copy()
if lineas_base is not None:
    for linea in lineas_base:
        x1, y1, x2, y2 = linea[0] if len(linea.shape) > 1 else linea
        cv2.line(resultado_base, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 4. Experimento: Líneas Cortas (Líneas Verdes)
lineas_exp2 = cv2.HoughLinesP(
    bordes,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=20,
    maxLineGap=10
)

resultado_exp2 = imagen.copy()
if lineas_exp2 is not None:
    for linea in lineas_exp2:
        x1, y1, x2, y2 = linea[0] if len(linea.shape) > 1 else linea
        cv2.line(resultado_exp2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Guardar resultado en carpeta local
dir_resultados = os.path.join(dir_script, "resultados")
os.makedirs(dir_resultados, exist_ok=True)
cv2.imwrite(os.path.join(dir_resultados, "ejemplo1_lineas_0133.jpg"), resultado_base)

# 5. Desplegar solo las 4 ventanas deseadas
cv2.imshow("1. Imagen Original (Loro)", imagen)
cv2.imshow("2. Bordes Canny", bordes)
cv2.imshow("3. Lineas Detectadas (Base - Rojas)", resultado_base)
cv2.imshow("4. Experimento (Lineas Cortas - Verdes)", resultado_exp2)

cv2.waitKey(0)
cv2.destroyAllWindows()