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
        
        self._crossVal(img, label, epochs=50)
        
        model = self._construirModelo()
        # Faz o teste
        model.fit(img, label, epochs=50)
        
        if not os.path.exists("modelos"):
            os.mkdir("modelos")
            
        # Salva o modelo
        model.save("modelos/libras_keras_model.keras")

    def _construirModelo(self):
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
    
    
    def _crossVal(self, img, label, n_splits=5, epochs=10):

        # Cria n folds para fazer o cross validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_no = 1
        accuracies = []

        for train_index, val_index in kf.split(img):
            
            X_train, X_val = img[train_index], img[val_index]
            y_train, y_val = label[train_index], label[val_index]

            model = self._construirModelo()
            model.summary()

            model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
            
            perda, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Acuracia no fold {fold_no}: {accuracy*100:.2f}%")
            accuracies.append(accuracy)

            fold_no += 1
            
        print(f"Acuracias por Fold: {[f'{acc*100:.2f}%' for acc in accuracies]}")
        print(f"Acuracia Media: {np.mean(accuracies)*100:.2f}%")
        print(f"Desvio Padrao da Acuracia: {np.std(accuracies)*100:.2f}%")


    
    def instanciar(self, img):
        model = load_model("modelos/libras_keras_model.keras")
        
        if (type(img) != np.ndarray):
           img = np.array(img)
        
        
        pred = model.predict(img.reshape(1,21,3), verbose=0)
        
        return chr(np.argmax(pred) + 96)
        