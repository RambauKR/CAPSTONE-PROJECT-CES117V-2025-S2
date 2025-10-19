import cv2
from flask import Flask, jsonify , render_template,url_for,Response, request

# Import processing classes
from processing.group01 import Group01Processor
from processing.group02 import Group02Processor
from processing.group03 import Group03Processor
from processing.group04 import Group04Processor

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

current_blur_classes = Group02Processor.DEFAULT_BLUR_CLASSES 

@app.route('/group01')
def group01():
    return render_template('group01.html')

@app.route('/group02')
def group02():
    return render_template('group02.html')

@app.route('/group03')
def group03():
    return render_template('group03.html')

@app.route('/group04')
def group04():
    return render_template('group04.html')

@app.route('/set_blur_classes', methods=['POST'])
def set_blur_classes():
    global current_blur_classes
    
    # 1. Check if the request body is valid JSON
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    
    # 2. Extract the classes array
    new_classes = data.get('classes', [])
    
    if not isinstance(new_classes, list):
        return jsonify({"error": "Classes field must be a list"}), 400

    # 3. Update the global setting
    current_blur_classes = new_classes
    
    # 4. Success Response (matches the JavaScript's expectation)
    print(f"Blur settings updated to: {current_blur_classes}")
    return jsonify({"message": "Settings updated successfully"}), 200


# Function to choose the processing based on route
def process_frame_based_on_route(route, frame):
    if route == 'group01':
        return Group01Processor.process(frame)
    elif route == 'group02':
        return Group02Processor.process(frame, blur_classes=current_blur_classes)
    elif route == 'group03':
        return Group03Processor.process(frame)
    elif route == 'group04':
        return Group04Processor.process(frame)
    else:
        # Default behavior for other routes
        return frame

def gen(route):
    cap = cv2.VideoCapture(0)

    try:
        while True:
           
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame based on the route
            processed_frame = process_frame_based_on_route(route, frame)

            # Encode the processed frame and return it as a byte stream
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()  # Release the camera when done


@app.route('/video_feed/<route>')
def video_feed(route):
    return Response(gen(route), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
