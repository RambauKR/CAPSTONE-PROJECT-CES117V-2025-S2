import cv2
from ultralytics import solutions
from collections import defaultdict

class Group03Processor:
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    vehicle_counts = defaultdict(int)
    
    # Initialize counter as class attribute
    try:
        counter = solutions.ObjectCounter(
            show=False,
            regions=[region_points],
            model="yolov8n.pt",
            tracker="bytetrack.yaml"
        )
    except Exception as e:
        print(f"Error initializing counter: {e}")
        counter = None

    @staticmethod
    def process(frame):
        """
        Process a single frame
        """
        if Group03Processor.counter is None:
            return frame
            
        try:
            results = Group03Processor.counter(frame)
            annotated_frame = results.plot()
            
            # Update counts if available
            if hasattr(results, 'counts') and results.counts:
                for cls_id, count in results.counts.items():
                    Group03Processor.vehicle_counts[cls_id] += count

            # Draw vehicle counts overlay
            y = 30
            for cls_id, count in Group03Processor.vehicle_counts.items():
                cv2.putText(
                    annotated_frame,
                    f"Class {cls_id}: {count}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y += 25

            return annotated_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

# Usage (static method):
# frame = Group03Processor.process(frame)