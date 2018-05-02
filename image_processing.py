# coding: utf8
from __future__ import division

import cv2
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


def process_camera():
    """
    Обработка изображения с камеры
    """
    cam = cv2.VideoCapture(0)
    img_prev = None
    while cam.isOpened():
        img = cam.read()[1]
        if camera_moved(img, img_prev):
            contours, main_contour, im_center, main_contour_points = process_image(img, cam_mode=True)
            if not (im_center is None):
                img_prev = img
            else:
                contours = None
                main_contour = None
                im_center = None
                main_contour_points = None
        draw_features(img, main_contour, im_center, main_contour_points)
        cv2.imshow('camera', img)
        button = cv2.waitKey(1)
        if button == 27:
            break
    cv2.destroyAllWindows()
    pass


def draw_features(img, main_contour, main_center, main_contour_points=None, contours=None):
    """
    Рисование найденных признаков (features) на изображении
    @param img: изображение
    @param main_contour: главный контур
    @param main_center: центр изображения
    @param main_contour_points: точки главного контура
    @param contours: контуры, найденные на изображении
    :return:
    """
    draw_point(img, main_center, (255, 255, 255))
    draw_contours(img, contours, (0, 255, 255), 1)
    draw_contours(img, main_contour, (255, 255, 255), 2)
    draw_contours(img, main_contour_points, (0, 0, 0), 1)


def draw_contours(img, contours, color, thickness):
    """
    Рисование контуров изображении
    @param img: изображение
    @param contours: контур
    @param color: цвет в формате RGB
    @param thickness: толщина
    :return:
    """
    if check_contour(contours):
        cv2.drawContours(img, contours, -1, color, thickness)


def camera_moved(img, img_prev):
    """
    Проверка движения камеры
    @param img: исходное изображение, полученное с камеры
    @param img_prev: предыдущее изображение, на котором был найден центр
    :return: True, если изображение камера двигалась, False - иначе
    """
    return camera_moved_descriptor_match(img, img_prev) or camera_moved_zeros(img, img_prev)


def camera_moved_descriptor_match(img, img_prev):
    """
    Метод, проверяющий, двигалась ли камера, на основе поиска совпадений дескрипторов
    https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
    @param img: исходное изображение, полученное с камеры
    @param img_prev: предыдущее изображение, на котором был найден центр
    :return: True, если изображение камера двигалась, False - иначе
    """
    if img_prev is None:
        return True
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(img_prev.copy(), cv2.COLOR_BGR2GRAY)
    img_descriptor = ORB.detectAndCompute(gray, None)[1]
    img_prev_descriptor = ORB.detectAndCompute(gray_prev, None)[1]
    if img_descriptor is None or img_prev_descriptor is None:
        return True
    matches = BF.match(img_descriptor, img_prev_descriptor)
    percent_of_match = float(len(matches) / len(img_descriptor))
    result = percent_of_match < DESCRIPTOR_MATCH
    return result


def camera_moved_zeros(img, img_prev):
    """
    Метод, проверяющий, двигалась ли камера, с помощью вычитания из изображения с камеры предыдущего изображения,
    на котором был найден центр.
    @param img: исходное изображение, полученное с камеры
    @param img_prev: предыдущее изображение, на котором был найден центр
    :return: True, если изображение камера двигалась, False - иначе
    """
    bitwise = np.subtract(img.copy(), img_prev.copy())
    nonzero = np.count_nonzero(bitwise)
    zero = img_prev.size - nonzero
    number_of_zeros = float(zero / img_prev.size)
    return number_of_zeros < ZEROS


def process_file(file_path):
    """
    Поиск центра на статичном изображении
    @param file_path: Путь к файлу
    :return:
    """
    img = cv2.imread(file_path)
    main_contour, main_center, main_contour_points = process_image(img)[1:]
    draw_features(img, main_contour, main_center, main_contour_points)
    cv2.imshow('image', img)
    button = cv2.waitKey(0)
    if button == 27:
        return
    pass


def convert_image(img):
    """
    Конвертация изображения для последующего поиска контуров на нем
    @param img: изображение
    :return: конвертированное изображение
    """
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
    img_out = floodfill_image_manual(img_out)

    return img_out


def floodfill_image_manual(img):
    """
    Заливка изображения по маске
    @param img: изображение
    :return: залитое изображение
    """
    morph = floodfill_image_morph(img.copy())
    # Маска, использующаяся для заливки (должна быть на 2 пикселя больше исходного изображения)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Заливаем изображение по маске
    floodfilled = morph.copy()
    cv2.floodFill(floodfilled, mask, (0, 0), 255)
    img_bitwise_not = cv2.bitwise_not(floodfilled)
    # Комбинируем, чтобы получить объекты переднего плана
    img_out = img | img_bitwise_not
    # cv2.imshow('morph', morph)
    # cv2.imshow('floodfilled', floodfilled)
    # cv2.imshow('bitwise not floodfilled', img_bitwise_not)
    # cv2.imshow('out', img_out)
    return img_out


def floodfill_image_morph(img):
    """
    Заполнение изображения с помощью морфологических операций OpenCV
    @param img: изображение
    :return: обработанное изображение
    """
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
    """
    Аппроксимирование контура
    @param contour: контур
    :return: аппроксимированный контур
    """
    epsilon = APPROX_EPS * cv2.arcLength(contour, closed=True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour


def find_center(contour):
    """
    Поиск центра контура на основе его моментов
    @param contour: контур
    :return: центр контура
    """
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        centroid_x = int(float(moments["m10"] / moments["m00"]))
        centroid_y = int(float(moments["m01"] / moments["m00"]))
        return centroid_x, centroid_y
    else:
        return None


def draw_point(img, point, color):
    """
    Рисование точки на изображении
    @param img: изображение
    @param point: точка (2D)
    @param color: цвет в формате RGB
    :return:
    """
    if point is not [] or point is not None:
        cv2.circle(img, point, 7, color, -1)


def find_centers(contours):
    """
    Поиск центров контуров
    @param contours: контуры
    :return: центры контуров
    """
    centers = np.array(filter(lambda x: not (x is None), map(find_center, contours)))
    return centers


def process_image(img, cam_mode=False):
    """
    Поиск центра на изображении
    @param img: изображение
    @param cam_mode: режим камеры
    """
    img_copy = img.copy()
    # Конвертируем изображение
    img_out = convert_image(img_copy)

    # Находим контуры
    contours = find_contours(img_out, cam_mode)

    # находим центры контуров
    centers = find_centers(contours)

    # находим выпуклую оболочку на основе центров
    main_contour_points = get_convex_hull(centers)

    # создаем контур из точек выпуклой оболочки
    main_contour = create_contour(main_contour_points)

    if len(extract_ndarray(main_contour)) < 2:
        return [None, None, None, None]

    # находим центр выпуклой оболочки
    main_center = find_center(extract_ndarray(main_contour))

    return [contours, main_contour, main_center, main_contour_points]


def check_contour(points):
    """
    Валидация множества точек
    @param points: множество точек
    :return: True, False
    """
    return points is not None and len(points) != 0 and len(extract_ndarray(points)) != 0


def filter_contour(contour):
    """
    Фильтрация контура относительно граничных значений
    @param contour: контур
    :return: True, если контур удовлетворяет условиям, False - иначе
    """
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return MIN_WIDTH < width < MAX_WIDTH and MAX_HEIGHT > height > MIN_HEIGHT


def create_contour(contour_points):
    """
    Объединение точек в контур
    @param contour_points: точки контура
    :return: контур
    """
    if contour_points is not None:
        return [np.array(map(extract_ndarray, contour_points), dtype=np.int32)]
    else:
        return [np.array([])]


def extract_ndarray(nparray):
    """
    Вспомогательный метод для извлечения ndarray из numpy array
    @param nparray: numpy array
    :return: ndarray
    """
    return nparray[0]


def get_convex_hull(points):
    """
    Получение выпуклой оболочки на основе множества точек
    @param points: точки (2D)
    :return: выпуклая оболочка
    """
    if check_contour(points):
        return cv2.convexHull(points)
    else:
        return np.array([])
