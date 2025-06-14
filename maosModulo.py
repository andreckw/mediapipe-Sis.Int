import cv2
import time
import numpy as np
import mediapipe as mp


class MaosDetector():
    
    def __init__(self, mode = False, maxMaos = 2, deteccao = 1, rastrear = 0.5):
        self.mode = mode
        self.maxMaos = maxMaos
        self.deteccao = deteccao
        self.rastrear = rastrear
        
        self.mpMaos = mp.solutions.hands
        self.maos = self.mpMaos.Hands(self.mode, self.maxMaos, self.deteccao, self.rastrear)
        
        self.mpDesenhar = mp.solutions.drawing_utils
        

    def encontrarMaos(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        self.resultados = self.maos.process(imgRGB)
        
        if self.resultados.multi_hand_landmarks:
            for maoLms in self.resultados.multi_hand_landmarks:
                if draw:
                    self.mpDesenhar.draw_landmarks(img, maoLms, self.mpMaos.HAND_CONNECTIONS)
        
        return img
    
    
    def encontrarPosicao(self, img, handNo=0, draw=True):
        
        lmList = []
        if self.resultados.multi_hand_landmarks:
            minhaMao = self.resultados.multi_hand_landmarks[handNo]
            for id, lm in enumerate(minhaMao.landmark):
                a, l, c = img.shape
                cx, cy = int(lm.x * l), int(lm.y * a)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        return lmList


if __name__ == "__main__":
    lCam = 640
    aCam = 480


    cam = cv2.VideoCapture(0)
    cam.set(3, lCam)
    cam.set(4, aCam)
    pTempo = 0

    detector = MaosDetector()

    while True:
        success, img = cam.read()
        img = detector.encontrarMaos(img)
        lmList = detector.encontrarPosicao(img, draw=False)
        
        if len(lmList) > 0:
           print(lmList)
        
        
        cTempo = time.time()
        fps = 1/(cTempo-pTempo)
        pTempo = cTempo
        
        cv2.putText(img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        
        cv2.imshow("Img", img)
        cv2.waitKey(100)