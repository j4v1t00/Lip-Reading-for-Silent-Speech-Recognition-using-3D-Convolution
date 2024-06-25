import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import joblib
import scikitplot as skplt

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
index_eyes = [159, 386]

carpeta_principal = 'data/img/'
vocales_info_b = []
img_lip_info_b = []

eyes_info = []
img_eyes_info = []
distance_eye = []

for entrada in os.listdir(carpeta_principal):
    ruta_absoluta = os.path.join(carpeta_principal, entrada)
    # Verifica si la entrada es una carpeta
    if os.path.isdir(ruta_absoluta):
        # Ruta de la carpeta que contiene las imágenes
        carpeta = 'data/img/' + entrada

    # Itera sobre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        # Verifica si el archivo es una imagen
        if archivo.endswith('.jpg') or archivo.endswith('.png') or archivo.endswith('.jpeg'):
            ruta_imagen = os.path.join(carpeta, archivo)
        
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
                lip_info = []
                lip_info_b = []
                eyes_info = []
                image = cv2.imread(ruta_imagen)
                height, width, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        for index in index_lips:
                            #lip_info.append(face_landmarks.landmark[index])
                            lip_info_b.append([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y])
                        for index in index_eyes:
                            eyes_info.append([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y])

                pos_x = []
                pos_y = []
                for i in lip_info_b:
                    pos_x.append(i[0])
                    pos_y.append(i[1])

                aux_x = np.mean(pos_x)
                aux_y = np.mean(pos_y)
                lip_info_b.append([aux_x, aux_y])
        
        img_lip_info_b.append(lip_info_b)
        img_eyes_info.append(eyes_info)
    vocales_info_b.append(img_lip_info_b)
    distance_eye.append(img_eyes_info)
    img_lip_info_b = []
    img_eyes_info = []
    
    
len_vi = len(vocales_info_b)
    
    
min = 1000
for i in vocales_info_b:
    if (len(i) < min) and (len(i) > 0):
        min = len(i)
        
for i in range(len(vocales_info_b)):
    vocales_info_b[i] = vocales_info_b[i][:min]
    distance_eye[i] = distance_eye[i][:min]
    
for i in vocales_info_b:
    print(len(i))
    
vocales_info_np_b = np.array(vocales_info_b)
distance_eye_np = np.array(distance_eye)


coordenada_central = 80
distancia_media_ojos = 63 #En milimetros
vocales_info_b_2 = []
img_lip_info_b_2 = []
for i in range(len(vocales_info_np_b)):
    for j in range(len(vocales_info_np_b[i])):
        lip_info_b_2 = []
        distancia_ojos = np.linalg.norm(distance_eye_np[i][j][0] - distance_eye_np[i][j][1])
        #print("distancia_ojos:", distancia_ojos)
        escala_ojos = distancia_media_ojos / distancia_ojos
        #print("escala_ojos:", escala_ojos)
        for k in range(len(vocales_info_np_b[i][j]) - 1):
            distancia = np.linalg.norm(vocales_info_np_b[i][j][k] - vocales_info_np_b[i][j][coordenada_central])
            #print("distancia:", distancia)
            #print("distancia final:", (distancia * escala_ojos))
            lip_info_b_2.append(distancia * escala_ojos)
        img_lip_info_b_2.append(lip_info_b_2)
    vocales_info_b_2.append(img_lip_info_b_2)
    img_lip_info_b_2 = []
    
    
longitud_array_vocals = min*len_vi
aux_a_v = 0
array_vocals = []
for i in range(longitud_array_vocals):
    if i % (longitud_array_vocals/5) == 0:
        aux_a_v += 1  
    array_vocals.append(aux_a_v)

df_array_vocals = pd.DataFrame(array_vocals, columns=['Y'])


df_2 = pd.DataFrame()
for i in range(len(vocales_info_b_2)):
    df_aux = pd.DataFrame(vocales_info_b_2[i])
    df_2 = pd.concat([df_2, df_aux])
    

# Definir los parámetros a ajustar
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}

# Realizar la búsqueda en cuadrícula para encontrar la mejor combinación de hiperparámetros
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(df_2, df_array_vocals.values.ravel())

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_2, df_array_vocals.values.ravel(), test_size=0.2, random_state=42)

# Crear un clasificador SVM
svm_classifier_2 = SVC(kernel='rbf', C=10, gamma=0.01)

# Entrenar el clasificador
svm_classifier_2.fit(X_train, y_train)

# Predecir las etiquetas de clase para el conjunto de prueba
y_pred = svm_classifier_2.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

skplt.metrics.plot_confusion_matrix(
    y_test, 
    y_pred, 
    normalize=True,
    title='Confusion Matrix SVM')

# Guardar el modelo
dump(svm_classifier_2, 'models/svm/svm_classifier_eyes.joblib')

# Realizar validación cruzada para evaluar el rendimiento del modelo SVM
scores = cross_val_score(SVC(), df_2, df_array_vocals.values.ravel(), cv=5)
print("Precisión de validación cruzada:", scores.mean())

X_train, X_test, y_train, y_test = train_test_split(df_2, df_array_vocals.values.ravel(), test_size=0.2, random_state=42)

rf_classifier_2 = RandomForestClassifier(n_estimators=1000, random_state=42)

rf_classifier_2.fit(X_train, y_train)

y_pred = rf_classifier_2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

skplt.metrics.plot_confusion_matrix(
    y_test, 
    y_pred, 
    normalize=True,
    title='Confusion Matrix Random Forest')

# Guardar el modelo
dump(rf_classifier_2, 'models/rf/rf_classifier_eyes.joblib')