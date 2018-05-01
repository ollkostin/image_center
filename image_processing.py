# coding: utf8
from __future__ import division

import cv2
import os
import numpy as np

MIN_WIDTH = 20
MIN_HEIGHT = 20
MAX_WIDTH = 400
MAX_HEIGHT = 400
APPROX_EPS = 0.001
DESCRIPTOR_MATCH = 0.63
ZEROS = 0.051
ORB = cv2.ORB_create()
BF = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


def process_camera(floodfill_func):
    cam = cv2.VideoCapture(0)
    img_prev = None
    main_contour = None
    main_center = None
    main_contour_points = None
    while cam.isOpened():
        img = cam.read()[1]
        if camera_moved(img, img_prev):
            main_contour, main_center, main_contour_points = process_image(img, floodfill_func, cam_mode=True)
            if (main_center != (0, 0)):
                img_prev = img
            else:
                main_contour = None
                main_center = None
                main_contour_points = None
        draw_features(img, main_contour, main_center, main_contour_points)
        cv2.imshow('camera', img)
        button = cv2.waitKey(1)
        if button == 27:
            break
    cv2.destroyAllWindows()


def draw_features(img, main_contour, main_center, main_contour_points):
    draw_point(img, main_center, (255, 255, 255))
    draw_contour(img, main_contour, (255, 255, 255), 2)
    draw_contour(img, main_contour_points, (0, 0, 0), 1)


def draw_contour(img, contour, color, thickness):
    if check_contour(contour):
        cv2.drawContours(img, contour, -1, color, thickness)


def camera_moved(img, img_prev):
    return camera_moved_descriptor_match(img, img_prev) or camera_moved_zeros(img, img_prev)


def camera_moved_descriptor_match(img, img_prev):
    """ Метод, проверяющий, двигалась ли камера. Используется BFMatcher для поиска совпадений дескрипторов

        https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html"""
    if img_prev is None:
        return True
    img_descriptor = ORB.detectAndCompute(img, None)[1]
    img_prev_descriptor = ORB.detectAndCompute(img_prev, None)[1]
    if img_descriptor is None or img_prev_descriptor is None:
        return True
    matches = BF.match(img_descriptor, img_prev_descriptor)
    percent_of_match = float(len(matches) / len(img_descriptor))
    result = percent_of_match < DESCRIPTOR_MATCH
    return result


def camera_moved_zeros(img, img_prev):
    bitwise = np.subtract(img_prev.copy(), img.copy())
    nonzero = np.count_nonzero(bitwise)
    zero = img_prev.size - nonzero
    number_of_zeros = float(zero / img_prev.size)
    return number_of_zeros < ZEROS


def cropped(img):
    h, w = img.shape[:2]
    h_half = int(h / 2)
    w_half = int(w / 2)
    lt = img[0:h_half, 0:w_half]
    lb = img[h_half:h, 0:w_half]
    rt = img[0:h_half, w_half:w]
    rb = img[h_half:h, w_half:w]
    return [lt, lb, rt, rb]


def process_file(file_path, floodfill_func, window_title=''):
    img = cv2.imread(file_path)
    main_contour, main_center, main_contour_points = process_image(img, floodfill_func)
    draw_features(img, main_contour, main_center, main_contour_points)
    cv2.imshow(window_title, img)
    button = cv2.waitKey(0)
    if button == 27:
        return


def convert_image(img, floodfill_func):
    img_copy = img.copy()
    # переводим в ч/б
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # находим грани с помощью алгоритма Кэнни
    img_out = cv2.Canny(img_copy, 150, 200)
    # Удаляем шумы билатеральным фильтром
    img_out = cv2.bilateralFilter(img_out, 9, 75, 75)

    # определяем пороговые значения
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    # заливка изображения, чтобы вычислить контуры
    img_out = floodfill_func(img_out)

    return img_out


def floodfill_image_manual(img):
    # Маска, использующаяся для заливки (должна быть на 2 пикселя больше исходного изображения)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Заливаем изображение по маске
    floodfilled = img.copy()
    cv2.floodFill(floodfilled, mask, (0, 0), 255)
    img_bitwise_not = cv2.bitwise_not(floodfilled)
    # Комбинируем, чтобы получить объекты переднего плана
    img_out = img | img_bitwise_not
    nonzero = cv2.countNonZero(img_out)
    if float(nonzero/img_out.size) > 0.9 :
        img_out = floodfill_image_morph(img)
    return img_out


def floodfill_image_morph(img):
    # kernel = np.array([[0, -1, 0], [-1, 4, 1], [0, -1, 0]], np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img_out = cv2.morphologyEx(img.copy(), cv2.MORPH_DILATE, kernel)
    return img_out


def find_contours(img_out, cam_mode=False):
    contours = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    if not cam_mode:
        contours = np.array(filter(filter_contour, contours))
    contours = map(approximate_contour, contours)
    return contours


def approximate_contour(contour):
    epsilon = APPROX_EPS * cv2.arcLength(contour, closed=True)
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


def process_image(img, floodfill_func, cam_mode=False):
    img_copy = img.copy()
    # Конвертируем изображение
    img_out = convert_image(img_copy, floodfill_func)

    # Находим контуры
    contours = find_contours(img_out, cam_mode)

    # находим центры контуров
    centers = find_centers(contours)

    # находим выпуклую оболочку на основе центров
    main_contour_points = get_convex_hull(centers)

    # создаем контур из точек выпуклой оболочки
    main_contour = create_contour(main_contour_points)

    if len(extract_element(main_contour)) < 2:
        return [None, None, None]

    # находим центр выпуклой оболочки
    main_center = find_center(extract_element(main_contour))

    return [main_contour, main_center, main_contour_points]


def check_contour(points):
    return points is not None and len(points) != 0 and len(extract_element(points)) != 0


def filter_contour(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return MIN_WIDTH < width < MAX_WIDTH and MAX_HEIGHT > height > MIN_HEIGHT


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
