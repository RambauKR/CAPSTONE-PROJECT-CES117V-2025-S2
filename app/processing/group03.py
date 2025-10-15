# processing/group03.py

import cv2

class Group03Processor:
    @staticmethod
    def process(frame):
        # Custom processing for group 03
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv
