#!/usr/bin/python
# coding: utf8

import os
import sys

import cv2
import image_processing as imp

camera_mode = '-c'
file_mode = '-f'


def main(argv):
    floodfill_func1 = imp.floodfill_image_manual
    floodfill_func2 = imp.floodfill_image_morph
    if argv[0] == camera_mode:
        imp.process_camera(floodfill_func2)
    elif argv[0] == file_mode:
        imp.process_file(argv, floodfill_func2, '1')
    pass



if __name__ == '__main__':
    print("OpenCV v{}".format(cv2.__version__))
    main(sys.argv[1:])
