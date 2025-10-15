import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

class Group03Processor:
    # YOLOv11 segmentation/tracking model
    model = YOLO("yolo11n.pt")
    
    # Tracking history
    track_history = defaultdict(lambda: deque(maxlen=30))
    
    # Counting line (y-coordinate)
    counting_line_y = 400
    
    # Keep track of counted IDs
    counted_ids = set()
    
    # Vehicle counts
    vehicle_counts = defaultdict(int)
    
    # Vehicle classes (COCO IDs)
    vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

    @staticmethod
    def process(frame):
        """
        Process a single frame:
        - Detect and track vehicles
        - Count vehicles crossing a line
        - Draw bounding boxes, center points, and counts
        """
        try:
            # Run YOLO tracking
            results = Group03Processor.model.track(
                frame, persist=True, classes=Group03Processor.vehicle_classes
            )

            annotated_frame = frame.copy()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                # Draw counting line
                cv2.line(
                    annotated_frame,
                    (0, Group03Processor.counting_line_y),
                    (frame.shape[1], Group03Processor.counting_line_y),
                    (0, 255, 0),
                    2,
                )

                # Process each detection
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    center_x, center_y = int(x), int(y)

                    # Draw center point
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Check if vehicle crosses the counting line
                    if (
                        center_y > Group03Processor.counting_line_y
                        and track_id not in Group03Processor.counted_ids
                    ):
                        Group03Processor.vehicle_counts[class_id] += 1
                        Group03Processor.counted_ids.add(track_id)
                        print(f"Vehicle {track_id} (class {class_id}) counted!")

                # Draw counts
                y_offset = 30
                for class_id, count in Group03Processor.vehicle_counts.items():
                    class_name = Group03Processor.get_class_name(class_id)
                    cv2.putText(
                        annotated_frame,
                        f"{class_name}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 25

                # Total count
                total = sum(Group03Processor.vehicle_counts.values())
                cv2.putText(
                    annotated_frame,
                    f"Total: {total}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            return annotated_frame

        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return frame

    @staticmethod
    def get_class_name(class_id):
        """Map COCO class IDs to readable names"""
        class_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }
        return class_names.get(class_id, f"Class {class_id}")
