import cv2
#computer vision for python, solve comp. vision problems,library
import numpy as np
import pygame

pygame.mixer.init()

sound = pygame.mixer.Sound(r"c:\Users\karos\OneDrive\Dokumenty\GitHub\Comp-Vision-Counter\Notification.Popup_01.wav")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

is_drinking = False

while True:
    ret,frame = cap.read()

    gray =  cv2.cvtcolor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h)in faces:
        mouth_roi = gray[y+h//2:y+h, x:x+w]
        _,mouth_thresh = cv2.threshold(mouth_roi,50,225,cv2.THRESH_BINARY)
        white_pixels = np.sum(mouth_thresh == 255)

        if white_pixels > 10000:
            if not is_drinking:
                print("Drinking water detected!")
                is_drinking = True
                sound.play()
        else:
            is_drinking = False


        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('frame',frame)



