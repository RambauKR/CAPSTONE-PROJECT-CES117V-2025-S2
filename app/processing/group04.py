import cv2
from ultralytics import solutions
import time
import csv
import os
from datetime import datetime

class Group04Processor:
    # Define region points for tracking
    region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
    regioncounter = solutions.RegionCounter(
        region=region_points,
        model="yolo11n-seg.pt",
        classes=[0]
    )

    analytics_data = []
    last_update_time = time.time()

    # CSV setup
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/group04_{datetime.now().strftime('%Y-%m-%d')}.csv"

    if not os.path.exists(csv_filename):
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "People_Count"])

    @staticmethod
    def process(frame):
        results = Group04Processor.regioncounter(frame)
        total_people = results.total_tracks

        current_time = time.time()
        if current_time - Group04Processor.last_update_time > 2:
            timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
            Group04Processor.analytics_data.append({"time": timestamp, "count": total_people})
            if len(Group04Processor.analytics_data) > 30:
                Group04Processor.analytics_data.pop(0)

            # Save to CSV
            with open(Group04Processor.csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, total_people])

            Group04Processor.last_update_time = current_time

        return results.plot_im
