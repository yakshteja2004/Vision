from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model

# ------------------ APP SETUP ------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ------------------ LOAD TRAINED MODEL ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "eye_disease_model.h5")

model = load_model(MODEL_PATH)

# ⚠️ IMPORTANT: class order MUST match training
CLASS_NAMES = ['cataract', 'glaucoma', 'normal', 'retina_disease']

# ------------------ PREDICTION FUNCTION ------------------
def predict_eye_disease(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_idx], confidence

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template('eye_analysis.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    # Save uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read image
    img = cv2.imread(file_path)

    # 🔥 DIRECT CNN PREDICTION (NO EYE DETECTION)
    disease, confidence = predict_eye_disease(img)
    result_text = f"{disease.capitalize()} ({confidence * 100:.2f}%)"

    # Save image (no rectangles needed for fundus images)
    processed_filename = 'processed_' + file.filename
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, img)

    return render_template(
        'eye_analysis.html',
        original_image=file.filename,
        processed_image=processed_filename,
        diagnosis=result_text,
        message="Prediction completed"
    )

# ------------------ IMAGE SERVING ------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)
