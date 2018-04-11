#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
from image_processing import *

CAMERA_MODE = '-c'
FILE_MODE = '-f'
window_title = 'image'

# TODO: try to use single collections instead of creation


def main(argv):
    if argv[0] == CAMERA_MODE:
        cam = cv2.VideoCapture(0)
        while cam.isOpened():
            img = cam.read()[1]
            img_out = process_image(img)
            cv2.imshow(window_title, img_out)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
    elif argv[0] == FILE_MODE:
        file_path = argv[1]
        if file_path is not None:
            if os.path.isfile(file_path):
                img = cv2.imread(argv[1])
                img = resize_if_necessary(img)
                img_out = process_image(img)
                cv2.imshow(window_title, img_out)
                cv2.waitKey(0)
        else:
            print("no file specified")
    pass


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
