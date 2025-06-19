import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import KFold

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

    def construirModelo(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(21,3)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(26),
        ])
        model.compile(optimizer="adam", 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      metrics=["accuracy"])
        return model
    
    
    def crossVal(self, img, label, n_splits=5, epochs=10):
        tf.random.set_seed(42)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_no = 1
        accuracies = []

        for train_index, val_index in kf.split(img):
            
            X_train, X_val = img[train_index], img[val_index]
            y_train, y_val = label[train_index], label[val_index]

            model = self.construirModelo()
            model.summary()

            model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
            
            perda, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Acuracia no fold {fold_no}: {accuracy*100:.2f}%")
            accuracies.append(accuracy)

            fold_no += 1
            
        print(f"Acuracias por Fold: {[f'{acc*100:.2f}%' for acc in accuracies]}")
        print(f"Acuracia Media: {np.mean(accuracies)*100:.2f}%")
        print(f"Desvio Padrao da Acuracia: {np.std(accuracies)*100:.2f}%")

        final_model = self.construirModelo()
        final_model.fit(img, label, epochs=epochs, verbose=0)
        
        if not os.path.exists("modelos"):
            os.mkdir("modelos")

        final_model.save("modelos/libras_keras_model_final.keras")

    
    def instanciar(self, img):
        model = load_model("modelos/libras_keras_model.keras")
        fmodel = load_model("modelos/libras_keras_model_final.keras")
        
        pred = model.predict(img.reshape(1,21,3))
        fpred = fmodel.predict(img.reshape(1,21,3))
        
        print(np.argmax(pred))
        print(np.argmax(fpred))
        