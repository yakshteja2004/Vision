from flask import Flask, render_template, request, send_from_directory, send_file
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from rag_chatbolt import load_pdf, ask_question
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------- APP SETUP ----------------
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "eye_disease_model.h5")
model = load_model(MODEL_PATH)

DEFECT_MODEL_PATH = os.path.join(BASE_DIR, "eye_defect_model.h5")
defect_model = load_model(DEFECT_MODEL_PATH)

CLASS_NAMES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

DEFECT_CLASSES = [
    "myopia",
    "hypermetropia",
    "presbyopia"
]

# ---------------- DISEASE PREDICTION ----------------
def predict_eye_disease(image):

    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_idx], confidence


# ---------------- DEFECT PREDICTION ----------------
def predict_eye_defect(image):

    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = defect_model.predict(img)

    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return DEFECT_CLASSES[class_idx], confidence


# ---------------- EYE POWER ESTIMATION ----------------
def estimate_eye_power(defect, confidence):

    if defect == "myopia":

        if confidence < 0.7:
            return "-1.00D to -2.00D", "Mild myopia – use weak concave lenses"

        elif confidence < 0.85:
            return "-2.00D to -4.00D", "Moderate myopia – use concave lenses"

        else:
            return "-4.00D to -6.00D", "Severe myopia – strong concave lenses"

    elif defect == "hypermetropia":

        if confidence < 0.7:
            return "+1.00D to +2.00D", "Mild hypermetropia – convex lenses"

        elif confidence < 0.85:
            return "+2.00D to +3.50D", "Moderate hypermetropia – convex lenses"

        else:
            return "+3.50D to +5.00D", "Severe hypermetropia – strong convex lenses"

    elif defect == "presbyopia":

        if confidence < 0.7:
            return "+1.00D", "Reading glasses recommended"

        elif confidence < 0.85:
            return "+1.50D to +2.00D", "Bifocal or reading glasses"

        else:
            return "+2.50D to +3.00D", "Strong reading glasses"

    else:
        return "No correction needed", "Normal vision"


# ---------------- REPORT LOGIC ----------------
def generate_report_details(disease, confidence):

    if disease == "normal":
        risk = "Low Risk"
        recommendation = "No immediate issues detected. Regular eye checkups are advised."
    else:
        risk = "High Risk"
        recommendation = "Potential eye disease detected. Please consult an ophthalmologist."

    return risk, recommendation


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("eye_analysis.html")


# ---------------- CHATBOLT PAGE ----------------
@app.route('/chatbolt')
def chatbolt():
    return render_template('chatbolt.html')


# ---------------- IMAGE ANALYSIS ----------------
@app.route('/analyze', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = cv2.imread(file_path)

    disease, confidence = predict_eye_disease(img)

    if disease == "normal":

        defect, defect_confidence = predict_eye_defect(img)

        power_range, correction = estimate_eye_power(defect, defect_confidence)

    else:

        defect = "None"
        power_range = "Not Applicable"
        correction = "Not Applicable"

    confidence_percent = f"{confidence * 100:.2f}%"

    risk_level, recommendation = generate_report_details(disease, confidence)

    processed_filename = 'processed_' + file.filename
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    cv2.imwrite(processed_path, img)

    return render_template(
        'reports.html',
        disease=disease.capitalize(),
        defect=defect.capitalize(),
        power_range=power_range,
        correction=correction,
        confidence=confidence_percent,
        risk_level=risk_level,
        recommendation=recommendation
    )


# ---------------- SERVE UPLOADED IMAGE ----------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ---------------- SERVE PROCESSED IMAGE ----------------
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


# ---------------- DOWNLOAD PDF REPORT ----------------
@app.route("/download-report", methods=["POST"])
def download_report():

    disease = request.form.get("disease", "Unknown")
    confidence = request.form.get("confidence", "Unknown")
    risk_level = request.form.get("risk_level", "Unknown")
    recommendation = request.form.get("recommendation", "No recommendation")

    defect = request.form.get("defect", "None")
    power_range = request.form.get("power_range", "Not Applicable")
    correction = request.form.get("correction", "Not Applicable")

    # -------- Patient Details --------
    patient_name = "Sample Patient"
    patient_id = "VG2026-001"
    age_group = "Adult"
    gender = "Male"
    eye_scanned = "Left Eye"

    from datetime import date
    scan_date = date.today().strftime("%d-%m-%Y")

    report_path = os.path.join(BASE_DIR, "eye_report_generated.pdf")

    styles = getSampleStyleSheet()
    content = []

    # TITLE
    content.append(Paragraph("Vision Guard - Eye Analysis Report", styles['Title']))
    content.append(Spacer(1,20))

    # -------- PATIENT DETAILS TABLE --------
    patient_data = [
        ["Patient Details", ""],
        ["Patient Name", patient_name],
        ["Patient ID", patient_id],
        ["Age Group", age_group],
        ["Gender", gender],
        ["Eye Scanned", eye_scanned],
        ["Scan Date", scan_date]
    ]

    patient_table = Table(patient_data, colWidths=[220,200])

    patient_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('SPAN',(0,0),(1,0)),
        ('ALIGN',(0,0),(1,0),'CENTER')
    ]))

    content.append(patient_table)
    content.append(Spacer(1,25))

    # -------- ANALYSIS SUMMARY --------
    summary_data = [
        ["Vision Guard Analysis Summary", ""],
        ["Detected Condition", disease],
        ["Confidence Level", confidence],
        ["Risk Level", risk_level],
        ["Vision Defect", defect],
        ["Estimated Eye Power", power_range],
        ["Suggested Correction", correction]
    ]

    summary_table = Table(summary_data, colWidths=[220,200])

    summary_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('SPAN',(0,0),(1,0)),
        ('ALIGN',(0,0),(1,0),'CENTER')
    ]))

    content.append(summary_table)
    content.append(Spacer(1,20))

    # -------- RECOMMENDATION --------
    content.append(Paragraph("Medical Recommendation", styles['Heading2']))
    content.append(Paragraph(recommendation, styles['Normal']))

    doc = SimpleDocTemplate(report_path)
    doc.build(content)

    return send_file(
        report_path,
        as_attachment=True,
        download_name="Vision_Guard_Report.pdf"
    )

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():

    if 'pdf_file' not in request.files:
        return "No file uploaded"

    file = request.files['pdf_file']

    if file.filename == '':
        return "No selected file"

    # Save file first
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Load PDF
    message = load_pdf(filepath)

    return message

# ---------------- CHATBOLT: ASK QUESTION ----------------
@app.route('/ask_chatbolt', methods=['POST'])
def ask_chatbolt():

    question = request.form['question']
    answer = ask_question(question)

    return answer


# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    app.run(debug=True)