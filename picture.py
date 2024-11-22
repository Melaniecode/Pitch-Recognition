import cv2
img = cv2.imread('sinker.jpeg')

if img is None:
    print("Error: Unable to load image. Check the file path.")
else:
    print(f"Image shape: {img.shape}")

    text = 'sinker'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2
    position = (50, 50)

    # 加文字到圖片中
    cv2.putText(img, text, position, font, font_scale, color, thickness)

    #顯示圖片
    cv2.imshow('Pitch Recognition', img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()








# 影像加入文字： https://steam.oxxostudio.tw/category/python/ai/opencv-text.html#a2