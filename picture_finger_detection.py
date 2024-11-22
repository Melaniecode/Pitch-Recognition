import cv2
import mediapipe as mp
import numpy as np

# 設定 Mediapipe Hand 模組
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=12)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)

# 指定圖片路徑
image_path = ""

# 讀取圖片
img = cv2.imread(image_path)
if img is None:
    print("圖片讀取失敗，請確認路徑是否正確。")
else:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    result = hands.process(imgRGB)

    imgHeight, imgWidth, _ = img.shape

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)
                cv2.putText(img, str(i), (xPos+10, yPos+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # 顯示結果
    cv2.imshow('Image Hand Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
