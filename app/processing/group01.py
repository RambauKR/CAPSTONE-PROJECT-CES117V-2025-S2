# processing/group01.py

import cv2

class Group01Processor:
    @staticmethod
    def process(frame):
        # Custom processing for group 01
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
