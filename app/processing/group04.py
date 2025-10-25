# processing/group04.py

import cv2
from ultralytics import solutions
import json



class Group04Processor:
    # Define region points
    region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
    regioncounter = solutions.RegionCounter(
        show=True,  # display the frame
        region=region_points,  # pass region points
        model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
        classes=[0]
    )

    @staticmethod
    def process(frame):
        # Custom processing for group 04
        results = Group04Processor.regioncounter(frame)
        return results.plot_im
    






    
