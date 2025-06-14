import cv2
import numpy as np
from maosModulo import MaosDetector
import time

# Instancia o detector de maos
detector = MaosDetector()

while True:
    # Pega a imagem 
    img = cv2.imread("fotos/a/1.png")
    
    # Redimensiona a imagem em 640x480 pixel(4:3)
    img = cv2.resize(img, (640, 480))
    
    # Procura as maos na imagem
    img = detector.encontrarMaos(img)
    
    # Procura os valores de cada ponto da imagem e mostra caso a lisata seha maior que 0
    lmList = detector.encontrarPosicao(img, draw=False)
    
    if len(lmList) > 0:
        print(lmList)
    
    # Mostra a imagem e espera 100ms
    cv2.imshow("Img", img)
    cv2.waitKey(100)