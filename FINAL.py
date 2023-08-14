import cv2
import numpy as np

vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, image = vc.read()
else:
    rval = False

rval, image = vc.read()
while rval:
    cv2.imshow("feed", image)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    image_threshold = cv2.bitwise_not(cv2.inRange(image_hsv, (0, 0, 86), (179, 110, 148)))  # 14,22 => 10,50
    contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("feed", image)

            # Count red pixels
            red_ct = 0
            for j in range(x, x+w, 5):
                for i in range(y, y+h, 5):
                    if (image_hsv[i, j, 0] >= 0 and image_hsv[i, j, 0] <= 10) or (image_hsv[i, j, 0] >= 160 and image_hsv[i, j, 0] <= 180):
                        red_ct += 1
            if red_ct / (w*h/25) > 0.7:
                print("RED")
            else:
                print("NOT RED")

    # Read webcam feed
    rval, image = vc.read()
    # Exit if 'q' pressed
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
