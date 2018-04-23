#!/usr/bin/python
# coding: utf8

import os
import sys

import cv2
from image_processing import *

camera_mode = '-c'
file_mode = '-f'

#TODO ошибка может быть
def main(argv):
    floodfill_func1 = floodfill_image_manual
    floodfill_func2 = floodfill_image_morph
    floodfill_func = floodfill_func2
    if argv[0] == camera_mode:
        process_camera(floodfill_func)
    elif argv[0] == file_mode:
        process_file(argv, floodfill_func1,'1')
        process_file(argv, floodfill_func2, '2')
    pass


def process_camera(floodfill_func):
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        img = cam.read()[1]
        #TODO: разобраться почему заново пересчитывается
        #TODO: разбить изображение на сетку и смотреть, как поменялось
        #TODO: применение, можно притянуть за уши

        img_out = process_image(img, floodfill_func)
        cv2.imshow('camera', img_out)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def process_file(argv, floodfill_func, window_title=''):
    file_path = argv[1]
    if file_path is not None:
        if os.path.isfile(file_path):
            img = cv2.imread(argv[1])
            img_out = process_image(img, floodfill_func)
            cv2.imshow(window_title, img_out)
            cv2.waitKey(0)
    else:
        print("no file specified")
        return


def resize_if_necessary(img):
    height, width = img.shape[:2]
    h_coef = float(height / height_max) if height > height_max else 1
    w_coef = float(width / width_max) if width > width_max else 1
    x = round(1 - float(1 / w_coef), 1)
    y = round(1 - float(1 / h_coef), 1)
    coef = max(x, y)
    if coef != 0.0:
        img = cv2.resize(img, (0, 0), fx=coef, fy=coef)
    return img


if __name__ == '__main__':
    print("OpenCV v{}".format(cv2.__version__))
    main(sys.argv[1:])
