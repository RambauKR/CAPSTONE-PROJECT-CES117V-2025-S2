import cv2
from flask import Flask, jsonify , render_template,url_for,Response, request

import threading
from collections import defaultdict
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import processing classes
from processing.group01 import Group01Processor
from processing.group02 import Group02Processor
from processing.group03 import Group03Processor
from processing.group04 import Group04Processor
import os

# Create processing directory if it doesn't exist
if not os.path.exists('processing'):
    os.makedirs('processing')


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

current_blur_classes = Group02Processor.DEFAULT_BLUR_CLASSES 

analytics_data = {
    "total_blurred_count": 0,
    "object_types": defaultdict(int),
    "heatmap_points": [],
}
analytics_lock = threading.Lock()

@app.route('/get_analytics')
def get_analytics():
    with analytics_lock:
        data = {"total_blurred_count": analytics_data["total_blurred_count"]}
    return jsonify(data)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', title='Analytics Dashboard')

@app.route('/charts')
def get_charts():
    with analytics_lock:
        obj_types = analytics_data["object_types"]
        if not obj_types:
            pie_fig_json = "{}"
        else:
            pie_fig = px.pie(
                names=list(obj_types.keys()),
                values=list(obj_types.values()),
                title='Blurred Object Categories',
                color_discrete_sequence=px.colors.sequential.Blues,
            )
            pie_fig.update_layout(
                template='plotly_dark', 
                legend_title_text='Object Types'
            )
            pie_fig_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    
        points = analytics_data["heatmap_points"]
        if not points:
            heatmap_fig_json = "{}"
        else:
            # Assuming a 640x480 frame for binning
            df = pd.DataFrame(points, columns=['x', 'y'])
            heatmap_fig = go.Figure(go.Histogram2d(
                x=df['x'],
                y=df['y'],
                colorscale='Jet',
                xbins=dict(start=0, end=640, size=20),
                ybins=dict(start=0, end=480, size=20),
                zauto=True
            ))
            heatmap_fig.update_layout(
                title='Blurred Region Heatmap',
                template='plotly_dark',
                xaxis=dict(title='X Coordinate', range=[0, 640]),
                yaxis=dict(title='Y Coordinate', range=[0, 480], autorange='reversed') 
            )
            heatmap_fig_json = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Keep heatmap points from growing indefinitely
            if len(analytics_data["heatmap_points"]) > 2000:
                analytics_data["heatmap_points"] = analytics_data["heatmap_points"][-1000:]

    return jsonify(pie_chart=pie_fig_json, heatmap_chart=heatmap_fig_json)


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

@app.route('/charts')
def charts():
    return render_template('charts.html')

@app.route('/set_blur_classes', methods=['POST'])
def set_blur_classes():
    global current_blur_classes
    
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    
    new_classes = data.get('classes', [])
    
    if not isinstance(new_classes, list):
        return jsonify({"error": "Classes field must be a list"}), 400

    current_blur_classes = new_classes
    
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
            processed_frame, frame_analytics = process_frame_based_on_route(route, frame)

            if frame_analytics:
                with analytics_lock:
                    analytics_data["total_blurred_count"] = frame_analytics["total_blurred_count"]
                    for key, value in frame_analytics["object_types"].items():
                        analytics_data["object_types"][key] = value
                    analytics_data["heatmap_points"].extend(frame_analytics["heatmap_points"])

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
