# coding: utf8
from __future__ import division

import cv2
import numpy as np

width_min = 20
height_min = 20
width_max = 400
height_max = 400
approx_eps = 0.001
descriptor_match = 0.63
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


def process_camera(floodfill_func):
    cam = cv2.VideoCapture(0)
    img_prev = None
    main_contour = None
    main_center = None
    main_contour_points = None
    while cam.isOpened():
        img = cam.read()[1]
        if camera_moved(img, img_prev):
            main_contour, main_center, main_contour_points, img_out = process_image(img, floodfill_func)
            img_prev = img
        draw_point(img, main_center, (255, 255, 255))
        draw_contour(img, main_contour, (255, 255, 255), 2)
        draw_contour(img, main_contour_points, (0, 0, 0), 1)
        cv2.imshow('camera', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def draw_contour(img, contour, color, thickness):
    if check_contour(contour):
        cv2.drawContours(img, contour, -1, color, thickness)


def camera_moved(img, img_prev):
    """ Метод, проверяющий, двигалась ли камера. Используется BFMatcher для поиска совпадений дескрипторов

        https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html"""
    if img_prev is None:
        return True
    img_descriptor = orb.detectAndCompute(img, None)[1]
    img_prev_descriptor = orb.detectAndCompute(img_prev, None)[1]
    matches = bf.match(img_descriptor, img_prev_descriptor)
    percent_of_match = float(len(matches) / len(img_descriptor))
    result = percent_of_match < descriptor_match
    return result


def is_moved(img, img_prev):
    bitwise = np.subtract(img_prev.copy(), img.copy())
    nonzero = np.count_nonzero(bitwise)
    zero = img_prev.size - nonzero
    number_of_zeros = float(zero / img_prev.size)
    result = number_of_zeros < 0.051
    print("{0}".format(number_of_zeros))
    return result


def cropped(img):
    h, w = img.shape[:2]
    h_half = int(h / 2)
    w_half = int(w / 2)
    lt = img[0:h_half, 0:w_half]
    lb = img[h_half:h, 0:w_half]
    rt = img[0:h_half, w_half:w]
    rb = img[h_half:h, w_half:w]
    return [lt, lb, rt, rb]


def process_file(argv, floodfill_func, window_title=''):
    file_path = argv[1]
    if file_path is not None and os.path.isfile(file_path):
        img = cv2.imread(argv[1])
        main_contour, main_center = process_image(img, floodfill_func)[:2]
        draw_point(img, main_center, (255, 255, 255))
        if check_contour(main_contour):
            cv2.drawContours(img, main_contour, -1, (255, 255, 255), 2)
        cv2.imshow(window_title, img)
        cv2.waitKey(0)
    else:
        print("no file specified")
        return


def convert_image(img, floodfill_func):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # находим грани с помощью алгоритма Кэнни
    img_out = cv2.Canny(img, 10, 200)

    # Удаляем шумы билатеральным фильтром
    img_out = cv2.bilateralFilter(img_out, 9, 75, 75)

    # определяем пороговые значения
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    # заливка изображения, чтобы вычислить контуры
    img_out = floodfill_func(img_out)

    return img_out


# TODO: Как заливать?
def floodfill_image_manual(img):
    img_copy = img.copy()
    # Маска, использующаяся для заливки (должна быть на 2 пикселя больше исходного изображения)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Заливаем
    cv2.floodFill(img_copy, mask, (0, 0), 255)
    img_bitwise_not = cv2.bitwise_not(img_copy)
    # Комбинируем, чтобы получить объекты переднего плана
    img_out = img | img_bitwise_not
    return img_out


# TODO : Другой вариант заливки
def floodfill_image_morph(img):
    kernel = np.array([[0, -1, 0], [-1, 4, 1], [0, -1, 0]], np.uint8)
    # kernel = np.ones((3, 3), np.uint8)
    img_out = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    return img_out


def find_contours(img_out):
    contours = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = np.array(filter(filter_contour, contours))
    contours = map(approximate_contour, contours)
    return contours


def approximate_contour(contour):
    epsilon = approx_eps * cv2.arcLength(contour, closed=True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour


def find_center(c):
    moments = cv2.moments(c)
    if moments["m00"] != 0:
        centroid_x = int(float(moments["m10"] / moments["m00"]))
        centroid_y = int(float(moments["m01"] / moments["m00"]))
        return centroid_x, centroid_y
    else:
        return 0, 0


def draw_centers(img, centers):
    color = [255, 0, 255]
    for x, y in centers:
        try:
            draw_point(img, (x, y), color)
        except TypeError:
            print("Type error occurred")
    pass


def draw_point(img, point, color):
    if point is not [] or point is not None:
        cv2.circle(img, point, 7, color, -1)


def find_centers(contours):
    centers = np.array(map(find_center, contours))
    return centers


def process_image(img, floodfill_func):
    img_copy = img.copy()
    # Конвертируем изображение
    img_out = convert_image(img_copy, floodfill_func)

    # Находим контуры
    contours = find_contours(img_out)

    # находим центры контуров
    centers = find_centers(contours)

    # находим выпуклую оболочку на основе центров
    main_contour_points = get_convex_hull(centers)

    # создаем контур из точек выпуклой оболочки
    main_contour = create_contour(main_contour_points)

    # находим центр выпуклой оболочки
    main_center = find_center(extract_element(main_contour))

    return [main_contour, main_center, main_contour_points, img_out]


def check_contour(points):
    return points is not None and len(points) != 0 and len(extract_element(points)) != 0


def filter_contour(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    # return width_min < width < width_max and height_max > height > height_min
    return True


def create_contour(contour_points):
    if contour_points is not None:
        return [np.array(map(extract_element, contour_points), dtype=np.int32)]
    else:
        return [np.array([])]


def extract_element(elements):
    return elements[0]


def get_convex_hull(points):
    if check_contour(points):
        return cv2.convexHull(points)
    else:
        return []
