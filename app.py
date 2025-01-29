import os
import cv2
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, Response
from yolo_inference import YOLOInference
import time

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(app.root_path, 'static', 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize YOLO model with your custom weights
yolo_infer = YOLOInference(
    model_path='C:/Code/Crowd-Management-System/runs/detect/train2/weights/best.pt'
)

@app.route('/')
def index():
    """Index page with an upload form and a link to live preview."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Set processed output path
    processed_filename = f"processed_{video_file.filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    # Run the YOLO processing in a background thread
    threading.Thread(
        target=yolo_infer.process_video,
        args=(video_path, processed_path),
        daemon=True  # daemon=True ensures thread won't block app shutdown
    ).start()

    # Redirect to a page that displays the live preview
    return redirect(url_for('live_preview'))

@app.route('/live_preview')
def live_preview():
    """Renders a template that embeds the MJPEG streaming route."""
    return render_template('live_preview.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            # If processing is done or hasn't started, break or wait
            if yolo_infer.latest_frame is None:
                time.sleep(0.1)
                continue

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', yolo_infer.latest_frame)
            if not ret:
                continue

            # Yield as an MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # small delay to avoid 100% CPU

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_video')
def process_video_route():
    input_video_path = "path_to_your_input.mp4"
    output_video_path = "path_to_your_output.mp4"

    # Start the YOLO process in a background thread
    t = threading.Thread(target=yolo_infer.process_video,
                         args=(input_video_path, output_video_path))
    t.start()
    return """
    <html>
      <body>
        <h1>Live Preview</h1>
        <img src="/video_feed" />
      </body>
    </html>
    """

if __name__ == '__main__':
    # IMPORTANT: set use_reloader=False to avoid double-threading issues in debug mode
    app.run(debug=True, use_reloader=False)
