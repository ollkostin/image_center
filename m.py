import cv2
import numpy as np

window_title = 'image from camera'
width_min = 30
height_min = 30
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
    contours = np.array(filter(filter_contour, contours))
    contours = map(approximate_contour, contours)
    centers = np.array(map(find_center, contours))
    return [contours, centers]


def approximate_contour(contour):
    epsilon = approx_eps * cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour


def find_center(c):
    moments = cv2.moments(c)
    if moments["m00"] != 0:
        centroid_x = int(float(moments["m10"] / moments["m00"]))
        centroid_y = int(float(moments["m01"] / moments["m00"]))
        return centroid_x, centroid_y
    pass


def draw_centers(img, centers):
    color = [255, 0, 255]
    for x, y in centers:
        try:
            draw_point(img, (x, y), color)
        except TypeError:
            print("Type error occurred")
    pass


def draw_point(img, point, color):
    cv2.circle(img, point, 7, color, -1)


def run():
    img = cv2.imread('picture.jpg')
    img_out = convert_image(img)
    contours, centers = find_contours_and_centers(img_out)
    main_contour_points = get_convex_hull(centers)
    _contour = [np.array(map(extract_point, main_contour_points), dtype=np.int32)]
    main_center = find_center(_contour[0])
    draw_point(img, main_center, (0, 0, 0))
    cv2.drawContours(img, _contour, 0, (255, 255, 255), 3)
    cv2.drawContours(img, main_contour_points, -1, (0, 0, 0), 3)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    pass


def filter_contour(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return width_min < width < width_max and height_max > height > height_min


def get_convex_hull(points):
    return cv2.convexHull(points)
    # return [np.array([[1, 1], [10, 50], [50, 50]], dtype=np.int32),
    #         np.array([[99, 99], [99, 60], [60, 99]], dtype=np.int32)]


def extract_point(point):
    return point[0]


if __name__ == '__main__':
    print("OpenCV v{}".format(cv2.__version__))
    run()
