import os
import cv2
import json
import feedparser
import numpy as np
import firebase_admin
from ultralytics import YOLO
from datetime import datetime
from keras.models import load_model
from firebase_admin import credentials, db
from flask import Flask, render_template, request, redirect, url_for

# Working Directory I
cwd = os.getcwd()
project_dir = f"{cwd}\Programs"

# Flask Configuration
app = Flask(__name__)

# Firebase Configuration
cred = credentials.Certificate(f"{project_dir}/firebase_credential_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'input_your_db_url_here'
})

# Directory Configuration II
UPLOAD_FOLDER   = f"{project_dir}\static/uploads"
CROP_FOLDER     = f"{project_dir}\static/crops"
HISTORY_FILE    = f"{project_dir}/history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AI Model
segment_model = YOLO(f"{project_dir}/best_skin.pt")

parts = ['lengan', 'paha', 'perut', 'telapak', 'wajah']
priority_order = ['wajah', 'perut', 'paha', 'lengan', 'telapak']
class_models = {part: load_model(f"{project_dir}/model_{part}.h5") for part in parts}
label_map = {0: "Normal", 1: "Jaundice"}

# Project Indicators
recommendations = {
    0: "Bayi dalam kondisi normal.",
    1: "Grade 1: Perlu observasi awal.",
    2: "Grade 2: Segera konsultasikan ke dokter.",
    3: "Grade 3: Butuh penanganan medis cepat.",
    4: "Grade 4: Kondisi serius, segera ke rumah sakit.",
    5: "Grade 5: Kondisi sangat kritis, penanganan darurat diperlukan!"
}

# Project Function I
def preprocess_crop(img, size=300):
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float16) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

# Project Function II
def save_uploaded_image(file):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return filepath

# Project Function III
def run_segmentation(image_rgb):
    return segment_model.predict(source=image_rgb, imgsz=512, conf=0.3, verbose=False)[0]

# Project Function IV
def analyze_parts(segment_result, image_rgb, file):
    results = {}
    part_predictions = {}
    detected_parts = []

    for box, cls in zip(segment_result.boxes.xyxy.cpu().numpy(), segment_result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls)
        if cls_id >= len(parts):
            continue

        part = parts[cls_id]
        crop = image_rgb[y1:y2, x1:x2]
        crop_path = os.path.join(CROP_FOLDER, f"{part}_{file.filename}")
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        inp = preprocess_crop(crop)
        pred = class_models[part].predict(inp)[0][0]
        label = label_map[int(pred > 0.5)]
        part_predictions[part] = int(pred > 0.5)
        detected_parts.append(part)
        results[part] = {"label": label, "image": os.path.basename(crop_path)}

    return results, part_predictions, detected_parts

# Project Function V
def calculate_grade(part_predictions, detected_parts):
    message = []
    final_grade = 0

    for i, part in enumerate(priority_order):
        if part not in detected_parts:
            message.append(f"Bagian {part} tidak terdeteksi, melanjutkan ke bawah...")
            continue

        prediction = part_predictions.get(part)
        if prediction is None:
            continue

        if prediction == 1:
            final_grade = i + 1
            continue
        else:
            if part == "wajah":
                final_grade = 0
                message.append("Wajah normal → bagian tubuh lainnya dianggap normal.")
                break
            elif part == "perut":
                message.append("Perut normal → bagian tubuh bawah dianggap normal.")
                break

    return final_grade, message

# Project Function VI
def save_to_firebase(username, record):
    try:
        ref = db.reference(username)
        ref.push(record)
    except Exception as e:
        print(f"Error while saving data into database! {e}")

# Project Function VII
def save_to_local(record):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump(record, f, indent=2)

# Project Function VIII
def get_latest_news(keyword="jaundice", max_articles=5):
    url = f"https://news.google.com/rss/search?q={keyword}&hl=id&gl=ID&ceid=ID:id"
    feed = feedparser.parse(url)
    news = []

    for entry in feed.entries[:max_articles]:
        news.append({
            "title": entry.title,
            "summary": entry.get("summary", ""),
            "link": entry.link
        })

    return news

# Project Function IX
def get_local_history(username):
    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    return data

# Project Function X
def get_db_history(username):
    try:
        db_ref = db.reference(username)
        history = db_ref.get()
        history_list = list(history.values())
    except:  
        history_list = []

    return history_list

# Apps Root
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Apps History
@app.route("/history", methods=["GET"])
def history():
    username = request.args.get("username", "").strip()
    history_data = get_db_history(username=username)
    return render_template("history.html", history=history_data)

# Apps Predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return redirect(url_for("index"))

    file = request.files["image"]
    username = request.form.get("username", "").strip()

    if not username:
        return redirect(url_for("index"))

    filepath = save_uploaded_image(file)
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segment_result = run_segmentation(image_rgb)
    if not segment_result.boxes:
        return render_template("result.html", results={}, grade="No Detection",
                               grade_text="Tidak Terdeteksi",
                               rekomendasi="Tidak ada bagian tubuh yang terdeteksi.",
                               message="")

    results, part_predictions, detected_parts = analyze_parts(segment_result, image_rgb, file)
    final_grade, message = calculate_grade(part_predictions, detected_parts)

    grade_text = f"Grade {final_grade} - {'Normal' if final_grade == 0 else 'Jaundice'}"
    grade_class = "low" if final_grade <= 1 else "medium" if final_grade <= 3 else "high"
    rekomendasi = recommendations[final_grade]

    new_record = {
        "filename": file.filename,
        "grade": grade_text,
        "grade_class": grade_class,
        "rekomendasi": rekomendasi,
        "results": results,
        "message": "<br>".join(message),
        "timestamp": datetime.now().strftime("%d %B %Y - %H:%M")
    }

    save_to_firebase(username, new_record)

    return render_template("result.html", results=results, grade=final_grade,
                           grade_text=grade_text, grade_class=grade_class,
                           rekomendasi=rekomendasi, message="<br>".join(message))

# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)