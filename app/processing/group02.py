# processing/group02.py

from ultralytics import YOLO
import cv2

import cv2

from ultralytics import YOLO
 
class Group02Processor:

    model = YOLO("yolo11n.pt") 
    BLUR_CLASSES = ['person', 'face', 'license_plate']
 
    @staticmethod
    def blur_object(frame, bbox, blur_strength=101, sigma=50):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), sigma)
        frame[y1:y2, x1:x2] = blurred_roi
        return frame
 
    @classmethod
    def process(cls, frame):
        results = cls.model(frame, verbose=False)
        processed_frame = frame.copy()
 
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = cls.model.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0]
 
                if label in cls.BLUR_CLASSES and conf > 0.5:
                    processed_frame = cls.blur_object(processed_frame, bbox)
 
        return processed_frame
 

    