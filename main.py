import cv2
from maosModulo import MaosDetector
from ia import Keras
import time

lCam = 640
aCam = 480


cam = cv2.VideoCapture(0)
cam.set(3, lCam)
cam.set(4, aCam)
pTempo = 0

detector = MaosDetector()
keras = Keras()
palavra = ""

while True:
    success, img = cam.read()
    img = detector.encontrarMaos(img)
    lmList = detector.encontrarPosicao(img, draw=False)
    
    if len(lmList) > 0:
        palavra = palavra + keras.instanciar(lmList)
    
    cv2.putText(img, palavra, (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    print(palavra)
    
    cTempo = time.time()
    fps = 1/(cTempo-pTempo)
    pTempo = cTempo
    
    cv2.imshow("Img", img)
    cv2.waitKey(100)
    
    if len(palavra) > 25:
        palavra = ""