import cv2
from flask import Flask, render_template,url_for,Response, request

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

# Function to choose the processing based on route
def process_frame_based_on_route(route, frame):
    if route == 'group01':
        return Group01Processor.process(frame)
    elif route == 'group02':
        return Group02Processor.process(frame)
    elif route == 'group03':
        return Group03Processor.process(frame)
    elif route == 'group04':
        return Group04Processor.process(frame)
    else:
        # Default behavior for other routes
        return frame

def gen(route):
    cap = cv2.VideoCapture(1)
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
