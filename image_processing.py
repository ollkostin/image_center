# coding: utf8

import cv2
import numpy as np

width_min = 20
height_min = 20
width_max = 400
height_max = 400
approx_eps = 0.001


def convert_image(img):
    # находим грани с помощью алгоритма Кэнни
    img_out = cv2.Canny(img, 10, 200)

    # Удаляем шумы билатеральным фильтром
    img_out = cv2.bilateralFilter(img_out, 9, 75, 75)

    # определяем пороговые значения
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

    # заливка изображения, чтобы вычислить контуры
    img_out = floodfill_image(img_out)
    return img_out


# TODO: Как заливать?
def floodfill_image(img):
    img_floodfill = img.copy()
    # Маска, использующаяся для заливки (должна быть на 2 пикселя больше исходного изображения)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Заливаем
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # cv2.imshow('flood', img_floodfill)
    img_bitwise_not = cv2.bitwise_not(img_floodfill)
    # cv2.imshow('bitwise', img_bitwise_not)
    # Комбинируем, чтобы получить объекты переднего плана
    img_out = img | img_bitwise_not
    # TODO : Другой вариант заливки
    # k = [[0, -1, 0], [-1, 4, 1], [0, -1, 0]]
    # kernel = np.array(k, np.uint8)
    # img_out = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return img_out


def find_contours(img_out):
    contours = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = np.array(filter(filter_contour, contours))
    contours = map(approximate_contour, contours)
    return contours


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
    if point is not [] or point is not None:
        cv2.circle(img, point, 7, color, -1)


def find_centers(contours):
    centers = np.array(map(find_center, contours))
    return centers


def process_image(img):
    img_copy = img.copy()
    # Конвертируем изображение
    img_out = convert_image(img_copy)

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

    draw_point(img_copy, main_center, (255, 255, 255))

    if check_contour(main_contour):
        cv2.drawContours(img_copy, main_contour, -1, (255, 255, 255), 2)

    if check_contour(main_contour_points):
        cv2.drawContours(img_copy, main_contour_points, -1, (0, 0, 0), 1)

    return img_copy


def check_contour(points):
    return points is not None and len(points) != 0 and len(extract_element(points)) != 0


def filter_contour(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return width_min < width < width_max and height_max > height > height_min


def create_contour(contour_points):
    if contour_points is not None:
        return [np.array(map(extract_element, contour_points), dtype=np.int32)]
    else:
        return [np.array([])]


def extract_element(point):
    return point[0]


def get_convex_hull(points):
    if check_contour(points):
        return cv2.convexHull(points)
    else:
        return []
