import os
import cv2
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, Response
from yolo_inference import YOLOInference

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(app.root_path, 'static', 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

yolo_infer = YOLOInference(model_path='C:/Code/Crowd-Management-System/runs/detect/train2/weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_heatmap', methods=['GET'])
def toggle_heatmap():
    yolo_infer.enable_heat_map = not yolo_infer.enable_heat_map
    print(f"[INFO] Heat map toggled to: {yolo_infer.enable_heat_map}")
    return redirect(url_for('live_preview'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return {"message": "No file found in request"}, 400
    video_file = request.files['video']
    if video_file.filename == '':
        return {"message": "No filename provided"}, 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    processed_filename = f"processed_{video_file.filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    threading.Thread(
        target=yolo_infer.process_video,
        args=(video_path, processed_path),
        daemon=True
    ).start()

    return {"message": "File uploaded successfully"}, 200

@app.route('/live_preview')
def live_preview():
    return render_template('live_preview.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if yolo_infer.latest_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', yolo_infer.latest_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_zoom', methods=['GET'])
def set_zoom():
    row = request.args.get('row', default=-1, type=int)
    col = request.args.get('col', default=-1, type=int)
    yolo_infer.set_zoom_cell(row, col)
    return {"status": "OK"}

@app.route('/zoom_feed')
def zoom_feed():
    """
    Streams the subimage from yolo_infer.get_zoomed_subimage().
    """
    def gen():
        while True:
            subimg = yolo_infer.get_zoomed_subimage()
            if subimg is None:
                # no cell or invalid subimage
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', subimg)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_video')
def process_video_route():
    input_video_path = "path_to_your_input.mp4"
    output_video_path = "path_to_your_output.mp4"
    threading.Thread(
        target=yolo_infer.process_video,
        args=(input_video_path, output_video_path),
        daemon=True
    ).start()
    return """
    <html>
      <body>
        <h1>Live Preview</h1>
        <img src="/video_feed" />
      </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
