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
        file_path = argv[1]
        if file_path is not None and os.path.isfile(file_path):
            imp.process_file(file_path, floodfill_func1, 'image')
        else:
            print("no file specified")
    pass


if __name__ == '__main__':
    print("OpenCV v{}".format(cv2.__version__))
    main(sys.argv[1:])
