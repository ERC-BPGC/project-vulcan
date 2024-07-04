import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

prev_x, direction = None, None
wave_count = 0
min_movement = 0.01
wave_timer = None
max_wave_time = 2

mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y) + abs(thumb_tip.z - index_tip.z)

            if distance > 0.17:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                if prev_x is None:
                    prev_x = x
                    continue

                if abs(x - prev_x) > min_movement:
                    new_direction = 'right' if x > prev_x else 'left'

                    if direction and new_direction != direction:
                        if wave_timer is None:
                            wave_timer = time.time()
                        else:
                            if time.time() - wave_timer <= max_wave_time:
                                wave_count+=1
                            else:
                                wave_count=1
                            wave_timer = time.time()

                        if wave_count >= 5:
                            print("Hand is waving")
                            wave_count = 0

                    direction = new_direction
                prev_x = x


hands.close()
cap.release()
