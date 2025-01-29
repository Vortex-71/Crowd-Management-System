import os
from flask import Flask, render_template, request, redirect, url_for, send_file, Response
from yolo_inference import YOLOInference

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(app.root_path, 'static', 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize YOLO model
yolo_infer = YOLOInference(model_path='yolov8m.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # Process video
    processed_filename = f"processed_{video_file.filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    yolo_infer.process_video(video_path, processed_path)

    # Redirect to results page
    return redirect(url_for('result', filename=processed_filename))

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/processed_video/<filename>')
def processed_video(filename):
    """
    Stream processed video in the browser instead of forcing download.
    """
    video_path = os.path.join(PROCESSED_FOLDER, filename)
    return Response(open(video_path, "rb"), mimetype="video/mp4")

if __name__ == '__main__':
    app.run(debug=True)
