import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import os

class Keras():
    
    def treinar(self, img, label):
        # Seta o random state para 42
        tf.random.set_seed(42)
        
        # Cria o modelo com 4 camadas
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(21,3)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(26),
        ])
        
        # Compila o modelo
        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        model.summary()
        
        # Faz o teste
        model.fit(img, label, epochs=10)
        
        if not os.path.exists("modelos"):
            os.mkdir("modelos")
            
        # Salva o modelo
        model.save("modelos/libras_keras_model.keras")
    
    
    def instanciar(self, img):
        model = load_model("modelos/libras_keras_model.keras")
        
        pred = model.predict(img.reshape(1,21,3))
        
        print(np.argmax(pred))
        