from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import base64, re, io, os
from PIL import Image, ImageOps
import tensorflow as tf
import os

# ================= APP =================
app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_PATH = "thai_digits_16_20.h5"
model = None

ADMIN_USER = "admin"
ADMIN_PASS = "1234"
LABEL_MAP = {
    0: "๑๖",
    1: "๑๗",
    2: "๑๘",
    3: "๑๙",
    4: "๒๐"
}

# ================= LOAD MODEL =================
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("MODEL LOADED SUCCESSFULLY")
    else:
        print("MODEL NOT FOUND")

load_model()

# ================= DEBUG =================
@app.route("/test")
def test():
    return "Flask OK"

# ================= PREPROCESS (FIXED FOR CANVAS) =================
def preprocess(img_b64):
    try:
        img_data = re.sub('^data:image/.+;base64,', '', img_b64)

        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")

        # 🔥 FIX 1: invert canvas (สำคัญมาก)
        img = ImageOps.invert(img)

        # 🔥 FIX 2: resize
        img = img.resize((64, 64))

        # 🔥 FIX 3: normalize
        img = np.array(img).astype("float32") / 255.0

        # 🔥 FIX 4: reshape
        img = img.reshape(1, 64, 64, 1)

        return img

    except Exception as e:
        print("PREPROCESS ERROR:", e)
        return None

# ================= FRONTEND =================
@app.route("/")
def home():
    return send_file("index.html")

@app.route("/login")
def login():
    return send_file("login.html")

@app.route("/admin")
def admin():
    return send_file("admin.html")

# ================= AUTH =================
@app.route("/auth", methods=["POST"])
def auth():
    data = request.json

    if data.get("user") == ADMIN_USER and data.get("pass") == ADMIN_PASS:
        return jsonify({"status": "ok"})

    return jsonify({"status": "fail"})

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        return jsonify({"error": "Model not loaded"})

    data = request.json

    if "image" not in data:
        return jsonify({"error": "No image received"})

    img = preprocess(data["image"])

    if img is None:
        return jsonify({"error": "Preprocess failed"})

    pred = model.predict(img, verbose=0)[0]

    idx = int(np.argmax(pred))
    conf = float(np.max(pred)) * 100

    # 🔥 convert 0–4 → 16–20
    result_number = LABEL_MAP.get(idx, idx)

    print("PREDICT:", idx, "=>", result_number, conf)

    return jsonify({
        "result": result_number,
        "raw_class": idx,
        "confidence": round(conf, 2),
        "debug_probs": pred.tolist()
    })

# ================= UPLOAD MODEL =================
@app.route("/upload", methods=["POST"])
def upload():
    global model

    if "model" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["model"]
    file.save(MODEL_PATH)

    load_model()

    return jsonify({"status": "model uploaded & reloaded"})

# ================= RUN =================
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    print(f"Server running on http://127.0.0.1:{port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )