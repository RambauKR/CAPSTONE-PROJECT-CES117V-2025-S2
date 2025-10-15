# processing/group02.py

import cv2

class Group02Processor:
    @staticmethod
    def process(frame):
        # Custom processing for group 02
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        return blur
