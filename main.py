import cv2
import numpy as np

# Ładowanie klasyfikatora Haar do detekcji twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

is_drinking = False

while True:
    # Odczyt klatki z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja obrazu na skale szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Region zainteresowania (ROI) dla ust
        mouth_roi = gray[y + h // 2:y + h, x:x + w]
        _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(mouth_thresh == 255)

        # Prosty próg wykrycia aktywności (np. picie wody)
        if white_pixels > 10000:
            if not is_drinking:
                print("Wykrywam Twoją twarz")
                is_drinking = True
        else:
            is_drinking = False

        # Rysowanie prostokąta wokół twarzy
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Wyświetlenie obrazu
    cv2.imshow('Frame', frame)

    # Wyjście po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
