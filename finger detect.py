import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Menggunakan MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    # Indeks landmark tangan sesuai dengan MediaPipe
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    fingers = []

    # Ibu jari
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Jari lainnya
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

# Menggunakan OpenCV untuk menangkap video dari kamera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Membalikkan frame secara horizontal untuk kesesuaian yang lebih alami
    frame = cv2.flip(frame, 1)

    # Konversi warna dari BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame dengan MediaPipe Hands
    results = hands.process(rgb_frame)

    # Menggambar anotasi tangan pada frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Menghitung jumlah jari yang terangkat
            num_fingers = count_fingers(hand_landmarks)

            # Menampilkan jumlah jari yang terangkat pada frame
            cv2.putText(frame, f'Driji Sing Kedeteksi: {num_fingers}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Pendeteksi Driji', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
        break

# Melepas objek VideoCapture dan menutup semua jendela
cap.release()
cv2.destroyAllWindows()
