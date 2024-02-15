import cv2
import math
import mediapipe as mp
from pynput.mouse import Button, Controller

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(1)
mouse = Controller()

while cap.isOpened():
    ret, frameunflip = cap.read()
    frame = cv2.flip(frameunflip, 1)
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for indexfinger, landmark in enumerate(hand_landmarks.landmark):

                #index finger positon
                if indexfinger == mp_hands.HandLandmark.INDEX_FINGER_TIP.value:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    #mouse move there
                    mouse.position = (x, y)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                finger_dist = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

                if finger_dist < 0.03:
                    cv2.putText(frame, "Touch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mouse.press(Button.left)
                    mouse.release(Button.left)

    #Display image
    cv2.imshow('Hand Tracking', frame)

    #exit the while loop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()