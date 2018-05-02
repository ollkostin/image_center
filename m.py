#!/usr/bin/python
# coding: utf8

import os
import sys

import cv2
import image_processing as imp

camera_mode = '-c'
file_mode = '-f'


def main(argv):
    if argv[0] == camera_mode:
        imp.process_camera()
    elif argv[0] == file_mode:
        file_path = argv[1]
        if file_path is not None and os.path.isfile(file_path):
            imp.process_file(file_path)
        else:
            print("no file specified")
    pass


if __name__ == '__main__':
    print("OpenCV v{}".format(cv2.__version__))
    main(sys.argv[1:])
