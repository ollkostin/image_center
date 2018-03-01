import cv2

print cv2.__version__

cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, img = cam.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgcanny = cv2.Canny(imggray, 100, 400)
    ret_, thresh = cv2.threshold(imgcanny, 140, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)
    cv2.imshow('cam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
