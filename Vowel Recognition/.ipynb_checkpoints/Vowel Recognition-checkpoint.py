import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
index_lips = [61, 76, 62, 78, 
              185, 184, 183, 191, 95, 96, 77, 146, 
              40, 74, 42, 80, 88, 89, 90, 91, 
              39, 73, 41, 81, 178, 179, 180, 181, 
              37, 72, 38, 82, 87, 86, 85, 84,
              0, 11, 12, 13, 14, 15, 16, 17,
              267, 302, 268, 312, 317, 316, 315, 314, 
              269, 303, 271, 311, 402, 403, 404, 405, 
              270, 304, 272, 310, 318, 319, 320, 321, 
              409, 408, 407, 415, 324, 325, 307, 375, 
              308, 292, 306, 291]

carpeta_principal = 'data/img/'
vocales_info = []
img_lip_info = []

for entrada in os.listdir(carpeta_principal):
    ruta_absoluta = os.path.join(carpeta_principal, entrada)
    # Verifica si la entrada es una carpeta
    if os.path.isdir(ruta_absoluta):
        # Ruta de la carpeta que contiene las imágenes
        carpeta = 'data/img/' + entrada

    # Itera sobre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        # Verifica si el archivo es una imagen (puedes ajustar esta condición según el tipo de imágenes que tengas)
        if archivo.endswith('.jpg') or archivo.endswith('.png') or archivo.endswith('.jpeg'):
            ruta_imagen = os.path.join(carpeta, archivo)
        
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
                lip_info = []
                image = cv2.imread(ruta_imagen)
                height, width, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        for index in index_lips:
                            #lip_info.append(face_landmarks.landmark[index])
                            lip_info.append([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y, face_landmarks.landmark[index].z])

                pos_x = []
                pos_y = []
                pos_z = []
                for i in lip_info:
                    pos_x.append(i[0])
                    pos_y.append(i[1])
                    pos_z.append(i[2])

                aux_x = np.mean(pos_x)
                aux_y = np.mean(pos_y)
                aux_z = np.mean(pos_z)
                lip_info.append([aux_x, aux_y, aux_z])
        
        img_lip_info.append(lip_info)
    vocales_info.append(img_lip_info)
    img_lip_info = []

aux_int = 1000
for i in vocales_info:
    if aux_int > len(i):
        min = len(i)
        
for i in range(len(vocales_info)):
    vocales_info[i] = vocales_info[i][:min]

vocales_info_np = np.array(vocales_info)

coordenada_central = 80
vocales_info_2 = []
img_lip_info_2 = []
for i in range(len(vocales_info_np)):
    for j in range(len(vocales_info_np[i])):
        lip_info_2 = []
        for k in range(len(vocales_info_np[i][j]) - 1):
            distancia = np.linalg.norm(vocales_info_np[i][j][k] - vocales_info_np[i][j][coordenada_central])
            lip_info_2.append(distancia)
        img_lip_info_2.append(lip_info_2)
    vocales_info_2.append(img_lip_info_2)
    img_lip_info_2 = []

longitud_array_vocals = 137*5
aux_a_v = 0

array_vocals = []
for i in range(longitud_array_vocals):
    if i % 137 == 0:
        aux_a_v += 1  
    array_vocals.append(aux_a_v)

df_array_vocals = pd.DataFrame(array_vocals, columns=['Y'])

df_a = pd.DataFrame(vocales_info_2[0])
df_e = pd.DataFrame(vocales_info_2[1])
df_i = pd.DataFrame(vocales_info_2[2])
df_o = pd.DataFrame(vocales_info_2[3])
df_u = pd.DataFrame(vocales_info_2[4])
df_2 = pd.concat([df_a, df_e, df_i, df_o, df_u])

# Definir los parámetros a ajustar
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}

# Realizar la búsqueda en cuadrícula para encontrar la mejor combinación de hiperparámetros
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(df_2, df_array_vocals.values.ravel())

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros para el SVM:", grid_search.best_params_)


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_2, df_array_vocals.values.ravel(), test_size=0.2, random_state=42)

# Crear un clasificador SVM
svm_classifier = SVC(kernel='rbf', C=100, gamma=1)

# Entrenar el clasificador
svm_classifier.fit(X_train, y_train)

# Predecir las etiquetas de clase para el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)


# Realizar validación cruzada para evaluar el rendimiento del modelo SVM
scores = cross_val_score(SVC(), df_2, df_array_vocals.values.ravel(), cv=5)
print("Precisión de validación cruzada:", scores.mean())


X_train, X_test, y_train, y_test = train_test_split(df_2, df_array_vocals.values.ravel(), test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy RF:", accuracy)