import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=12)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)
while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        result = hands.process(imgRGB)

        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    cv2.putText(img, str(i), (xPos+10, yPos+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        cv2.imshow('Pitch Recognition', cv2.flip(img, 1))

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()