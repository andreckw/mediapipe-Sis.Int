import cv2
import numpy as np
import os
from maosModulo import MaosDetector
from ia import Keras

# Instancia o detector de maos
detector = MaosDetector()

imagens = []
labels = []

for pasta in os.listdir("fotos"):
    
    
    for arquivo in os.listdir(f"fotos/{pasta}"):
        caminho = f"fotos/{pasta}/{arquivo}"
            
        # Pega a imagem 
        img = cv2.imread(caminho)
        
        # Redimensiona a imagem em 640x480 pixel(4:3)
        img = cv2.resize(img, (640, 480))
        
        # Procura as maos na imagem
        img = detector.encontrarMaos(img)
        
        # Procura os valores de cada ponto da imagem e mostra caso a lisata seha maior que 0
        lmList = detector.encontrarPosicao(img, draw=False)
        
        if len(lmList) > 0:
            # Pega os valores dos pontos
            imagens.append(lmList)
            # Pega a letra, transforma em ascci e dimini por 96 para ficar entre 1 a 25
            labels.append(int(ord(pasta)) - 96)
                
            
        
        # Mostra a imagem e espera 1000ms
        cv2.imshow("Img", img)
        cv2.waitKey(1)
    
imagens = np.array(imagens)
labels = np.array(labels)

keras = Keras()
keras.treinar(img=imagens, label=labels)

keras.instanciar(imagens[7])
print(labels[7])

