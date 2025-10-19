# processing/group02.py

from ultralytics import YOLO
import cv2

class Group02Processor:

    model = YOLO("yolo11n.pt") 
    
    # Default classes to blur if none are provided by the application settings.
    DEFAULT_BLUR_CLASSES = ['dog']
 
    @staticmethod
    def blur_object(frame, bbox, blur_strength=101, sigma=50):
       
        # Ensure blur_strength is an odd number
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounding box coordinates are within frame bounds
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract the Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2]
        
        # Skip if ROI is invalid or empty
        if roi.size == 0 or roi.shape[0] <= 1 or roi.shape[1] <= 1:
            return frame

        # Apply Gaussian blur
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), sigma)
        
        # Place the blurred ROI back into the frame
        frame[y1:y2, x1:x2] = blurred_roi
        return frame
 
    @classmethod
    def process(cls, frame, blur_classes=None):
    
        classes_to_process = blur_classes 
        
        processed_frame = frame.copy()
        
        # Return frame immediately if there's nothing to blur
        if not classes_to_process:
            return processed_frame
        
   
        results = cls.model(frame, verbose=False)
        
        print(classes_to_process)

        for result in results:
            # Iterate through all detected bounding boxes
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = cls.model.names.get(cls_id)
                conf = float(box.conf[0])
                bbox = box.xyxy[0]
 
                if label in classes_to_process and conf > 0.5:
                    processed_frame = cls.blur_object(processed_frame, bbox)
 
        return processed_frame
