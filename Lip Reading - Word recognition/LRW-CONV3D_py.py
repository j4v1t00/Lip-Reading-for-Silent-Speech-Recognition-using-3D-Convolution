import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scikitplot as skplt

from sklearn.model_selection import train_test_split


carpeta_principal = 'lipread_mp4_demo/'
word_path = []
ttv_path = []

for entrada in os.listdir(carpeta_principal):
    ruta_absoluta = os.path.join(carpeta_principal, entrada)
    # Verifica si la entrada es una carpeta
    if os.path.isdir(ruta_absoluta):
        # Ruta de la carpeta que contiene las imágenes
        carpeta_1 = carpeta_principal + entrada
        # Itera sobre todos los archivos en la carpeta
        for subcarpeta in os.listdir(carpeta_1):
            # Itera sobre todas las carpetas test, train y val
            carpeta_2 = carpeta_1+'/'+subcarpeta
            # Obtener la lista de elementos (archivos y carpetas) en la carpeta
            elementos = os.listdir(carpeta_2)
            # Contar el número de elementos en la lista
            num_elementos = len(elementos)
            ttv_path.append(int(num_elementos/2))

        word_path.append(ttv_path)
        ttv_path = []

word_path

carpeta_principal = 'img_mouth_lipread_resize_demo_2/'

ttv_cont = 0
ttv_value = []
for files in word_path:
    for file_lenght in files:
        for i in range(file_lenght):
            ttv_value.append(ttv_cont)
        ttv_cont = ttv_cont + 1
    ttv_cont = 0
    

test = []
train = []
value = []
clip = []
i = 0

for entrada in os.listdir(carpeta_principal):
    ruta_absoluta = os.path.join(carpeta_principal, entrada)
    image= cv2.imread(ruta_absoluta)
    clip.append(image)
    
    if(len(clip) == 10):
        if (ttv_value[i] == 0):
            test.append(clip)
        else:
            if (ttv_value[i] == 1):
                train.append(clip)
            else:
                if (ttv_value[i] == 2):
                    value.append(clip)
        i = i + 1
        clip = []
        
        
test_np = np.array(test)
train_np = np.array(train)
value_np = np.array(value)
dataset_np = np.concatenate((train_np, test_np), axis=0)

y_test = []
y_train = []
y_value = []
word_count = 0
for files in word_path:
    for i in range(files[0]):
        y_test.append(word_count)
    for i in range(files[1]):
        y_train.append(word_count)
    for i in range(files[2]):
        y_value.append(word_count)
    word_count = word_count + 1
    
y_test_np = np.array(y_test)
y_train_np = np.array(y_train)
y_value_np = np.array(y_value)
y_dataset_np = np.concatenate((y_train_np, y_test_np), axis=0)

X_train, X_test, y_train, y_test = train_test_split(dataset_np, y_dataset_np, test_size=0.2, random_state=42)

class CFG:
    epochs = 10
    batch_size = 32
    classes = ["ATTACK", "BLACK", "FINAL", "IMPACT", "LATER", "MEDIA", "OFFICE", "PRESS", "SPEND", "WEEKS"]
    
train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(CFG.batch_size * 4).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)
valid = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(10, 50, 100, 3)),
    tf.keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Conv3D(64, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Conv3D(128, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling3D(),
    tf.keras.layers.Dense(len(CFG.classes), activation="softmax")
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=[
        "accuracy"
    ]
)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/model_PreMouth_Resize_2.keras", 
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)
history = model.fit(
    train,
    epochs=30, 
    validation_data=valid, 
    callbacks=[checkpoint]
)

model.load_weights("model_PreMouth_Resize_2.keras")

for metrics in [("loss", "val_loss"), ("accuracy", "val_accuracy")]:
    pd.DataFrame(history.history, columns=metrics).plot()
    plt.show()
    
# Guardar el modelo
model.save('models/LRW_Conv3D_resize_2.keras')

# Cargar el modelo desde el formato SavedModel
model = tf.keras.models.load_model('models/LRW_Conv3D_resize_2.keras')

# Mostrar el resumen del modelo para verificar que se ha cargado correctamente
model.summary()

# Obtener las predicciones del modelo
y_pred_prob = model.predict(X_train)
# Convertir las predicciones de probabilidad a etiquetas
y_pred = np.argmax(y_pred_prob, axis=1)

skplt.metrics.plot_confusion_matrix(
    y_train, 
    y_pred,
    figsize=(12, 12),  
    title='Matriu de Confusió CONV3D',
    cmap='Blues',  
    normalize=True
)

# Mostrar los números de cada casilla
plt.show()

