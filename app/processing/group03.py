import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np

class Group03Processor:
    model = YOLO("yolo11x.pt")
    track_history = defaultdict(lambda: deque(maxlen=30))
    vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

    # 6 lane regions for 1920x1080
    # lane_regions = {
    # "lane_1": [(0, 0), (320, 0), (320, 1080), (0, 1080)],
    # "lane_2": [(320, 0), (640, 0), (640, 1080), (320, 1080)],
    # "lane_3": [(640, 0), (960, 0), (960, 1080), (640, 1080)],
    # "lane_4": [(960, 0), (1280, 0), (1280, 1080), (960, 1080)],
    # "lane_5": [(1280, 0), (1600, 0), (1600, 1080), (1280, 1080)],
    # "lane_6": [(1600, 0), (1920, 0), (1920, 1080), (1600, 1080)],
    # }

    # 6 lane regions for 1920x1080
    lane_regions = {
    # Left carriageway (moving away from camera), from far-left shoulder toward median
    "lane_1": [(650, 300), (750, 300), (260, 650), (0, 650)],   # leftmost
    "lane_2": [(750, 300), (820, 300), (500, 650), (260, 650)],
    "lane_3": [(820, 300), (880, 300), (800, 650), (500, 650)],   # next to median

    # Right carriageway (moving toward camera), from median toward far-right shoulder
    "lane_4": [(970, 300), (1030, 300), (1300, 650), (1100, 650)],  # next to median
    "lane_5": [(1030, 300), (1080, 300), (1500, 650), (1300, 650)],
    "lane_6": [(1080, 300), (1150, 300), (1700, 650), (1500, 650)], # rightmost
}
 

    lane_counts_in = defaultdict(lambda: defaultdict(int))
    lane_counts_out = defaultdict(lambda: defaultdict(int))
    counted_ids_in = set()
    counted_ids_out = set()

    @staticmethod
    def point_in_polygon(point, polygon):
        x, y = point
        poly = cv2.convexHull(cv2.UMat(np.array(polygon, np.int32))).get()
        return cv2.pointPolygonTest(poly, (x, y), False) >= 0

    @staticmethod
    def process(frame):
        try:
            results = Group03Processor.model.track(
                frame, persist=True, classes=Group03Processor.vehicle_classes
            )
            annotated_frame = frame.copy()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    center_x, center_y = int(x), int(y)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    Group03Processor.track_history[track_id].append(center_y)

                    lane_name = None
                    for lane, polygon in Group03Processor.lane_regions.items():
                        if Group03Processor.point_in_polygon((center_x, center_y), polygon):
                            lane_name = lane
                            break
                    if lane_name is None:
                        continue

                    # Determine direction
                    if len(Group03Processor.track_history[track_id]) >= 2:
                        prev_y = Group03Processor.track_history[track_id][-2]
                        curr_y = center_y

                        if prev_y < 540 <= curr_y and track_id not in Group03Processor.counted_ids_in:
                            Group03Processor.lane_counts_in[lane_name][class_id] += 1
                            Group03Processor.counted_ids_in.add(track_id)
                        elif prev_y > 540 >= curr_y and track_id not in Group03Processor.counted_ids_out:
                            Group03Processor.lane_counts_out[lane_name][class_id] += 1
                            Group03Processor.counted_ids_out.add(track_id)

            # Draw lane boundaries and counts
            for lane, polygon in Group03Processor.lane_regions.items():
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)

                # IN counts (left side of lane)
                x_in = min([p[0] for p in polygon]) + 20
                y_offset = 50
                for class_id, count in Group03Processor.lane_counts_in[lane].items():
                    cv2.putText(
                        annotated_frame,
                        f"{lane} IN {Group03Processor.get_class_name(class_id)}:{count}",
                        (x_in, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 40

                # OUT counts (right side of lane)
                x_out = max([p[0] for p in polygon]) - 300  # adjust offset for wide text
                y_offset = 50
                for class_id, count in Group03Processor.lane_counts_out[lane].items():
                    cv2.putText(
                        annotated_frame,
                        f"{lane} OUT {Group03Processor.get_class_name(class_id)}:{count}",
                        (x_out, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    y_offset += 40

            return annotated_frame

        except Exception as e:
            print(f"Error: {e}")
            return frame

    @staticmethod
    def get_class_name(class_id):
        class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        return class_names.get(class_id, f"Class {class_id}")
