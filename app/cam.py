import cv2

def find_available_camera_index():
    index = 0
    available_indexes = []
    
    while index < 10:  # Check up to 10 camera indexes
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            available_indexes.append(index)
            cap.release()
        index += 1
    
    if not available_indexes:
        print("No available cameras found.")
    return available_indexes

# Check for available cameras
available_cameras = find_available_camera_index()

if available_cameras:
    print(f"Available camera indexes: {available_cameras}")
else:
    print("No cameras available.")