import cv2
import numpy as np

window_title = 'image from camera'
width_min = 50
height_min = 50
width_max = 400
height_max = 400
approx_eps = 0.001


def convert_image(img):
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_out = cv2.Canny(img_out, 100, 200)
    # remove noise
    img_out = cv2.bilateralFilter(img_out, 9, 75, 75)
    # img_thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)[1]
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    img_out = floodfill_image(img_out)
    return img_out


def floodfill_image(img_thresh):
    img_floodfill = img_thresh.copy()
    # Mask used to flood filling. Size needs to be 2 pixels more than the image.
    h, w = img_thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0) by mask
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # Combine the two images to get the foreground
    img_out = img_thresh | cv2.bitwise_not(img_floodfill)
    return img_out


def find_contours_and_centers(img_out):
    contours = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = filter(filter_contour, contours)
    contours = map(approximate_contour, contours)
    centers = map(find_center, contours)
    return [contours, centers]


def approximate_contour(contour):
    epsilon = approx_eps * cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour


def box_cnt(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def find_center(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        centroid_x = int(float(M["m10"] / M["m00"]))
        centroid_y = int(float(M["m01"] / M["m00"]))
        return centroid_x, centroid_y
    pass


def draw_centers(img, centers):
    for c in centers:
        try:
            cv2.circle(img, (c[0], c[1]), 7, (255, 255, 255), -1)
        except TypeError:
            print("Type error")
    pass


def __main__():
    print cv2.__version__
    img = cv2.imread('pic.png')
    img_out = convert_image(img)
    contours, centers = find_contours_and_centers(img_out)
    # TODO: remove internal points and build contour. Then find center
    # main_contour = np.array([centers],dtype=np.int32)
    # centers.append(
    #     find_center(main_contour)
    # )
    cv2.drawContours(img, main_contour, -1, (0, 255, 0), 3)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    draw_centers(img, centers)
    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    pass


def filter_contour(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return width_min < width < width_max and height_max > height > height_min


__main__()

# contours = [np.array([[1, 1], [10, 50], [50, 50]], dtype=np.int32),
#             np.array([[99, 99], [99, 60], [60, 99]], dtype=np.int32)]
