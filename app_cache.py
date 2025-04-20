import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from ultralytics import YOLO
from datetime import datetime
import feedparser

app = Flask(__name__)

cwd = os.getcwd()

UPLOAD_FOLDER = "static/uploads"
CROP_FOLDER = "static/crops"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

segment_model = YOLO(f"{cwd}\Programs/best.pt")

parts = ['lengan', 'paha', 'perut', 'telapak', 'wajah']
priority_order = ['wajah', 'perut', 'paha', 'lengan', 'telapak']
class_models = {part: load_model(f"model_{part}.h5") for part in parts}
label_map = {0: "Normal", 1: "Jaundice"}

recommendations = {
    0: "Bayi dalam kondisi normal.",
    1: "Grade 1: Perlu observasi awal.",
    2: "Grade 2: Segera konsultasikan ke dokter.",
    3: "Grade 3: Butuh penanganan medis cepat.",
    4: "Grade 4: Kondisi serius, segera ke rumah sakit.",
    5: "Grade 5: Kondisi sangat kritis, penanganan darurat diperlukan!"
}

def preprocess_crop(img, size=300):
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float16) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/history", methods=["GET"])
def history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    return render_template("history.html", history=data[::-1])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    results = {}
    message = []
    part_predictions = {}
    detected_parts = []

    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segment_result = segment_model.predict(source=image_rgb, imgsz=512, conf=0.3, verbose=False)[0]

    if not segment_result.boxes:
        return render_template("result.html", results={}, grade="No Detection",
                               grade_text="Tidak Terdeteksi",
                               rekomendasi="Tidak ada bagian tubuh yang terdeteksi.",
                               message="")

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

    final_grade = 0
    message = []

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

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(new_record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return render_template("result.html", results=results, grade=final_grade,
                           grade_text=grade_text, grade_class=grade_class,
                           rekomendasi=rekomendasi, message="<br>".join(message))


if __name__ == "__main__":
    app.run(debug=True)
