# processing/group04.py

import cv2

class Group04Processor:
    @staticmethod
    def process(frame):
        # Custom processing for group 04
        resized_frame = cv2.resize(frame, (640, 480))
        edges = cv2.Canny(resized_frame, 50, 150)
        return edges
