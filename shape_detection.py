import cv2
import numpy as np
import pytesseract
import imutils
import threading
import datetime
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 76, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

while True:
    _, frame = cap.read()
    resize =cv2.resize(frame,(640,480))

    """
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)#gri filtre uygula
    noise = cv2.bilateralFilter(gray, 10, 50, 50)#keskin yerleri yumuşat
    kernel = np.array([[1,1,1], [1,-8,1], [1,1,1]])
    kernel = cv2.filter2D(noise, -1, kernel)
    equal_histogram = cv2.equalizeHist(kernel)#siyah beyaz renkleri keskinleştirme yapıldı
    """




    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 400:
            cv2.drawContours(resize, [approx], 0, (0, 0, 0), 5)
            if len(approx) == 4:
                mask= np.ones(resize.shape,np.uint8)
                new_img = cv2.drawContours(mask, [approx], 0, (0, 0, 0), 1)
                new_img = cv2.bitwise_and(resize, resize, mask=mask)
                cv2.imwrite("as.jpg",new_img)
                print("buldum.................")


    cv2.imshow("Frame", resize)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
