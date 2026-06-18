from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ── Paths ──
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'ml_models', 'saved_models')

# ── Load Crop Model ──
crop_model  = joblib.load(os.path.join(MODEL_DIR, 'crop_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
crop_le     = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'model_meta.json')) as f:
    crop_meta = json.load(f)

# ── Load Disease Model ──
dis_model  = joblib.load(os.path.join(MODEL_DIR, 'disease_model.pkl'))
dis_scaler = joblib.load(os.path.join(MODEL_DIR, 'disease_scaler.pkl'))
dis_le     = joblib.load(os.path.join(MODEL_DIR, 'disease_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'disease_meta.json')) as f:
    dis_meta = json.load(f)

REMEDIES = dis_meta['remedies']

print(f"✅ Crop Model loaded     — Accuracy: {crop_meta['accuracy']}%")
print(f"✅ Disease Model loaded  — Accuracy: {dis_meta['accuracy']}%")

# ── Crop Info ──
CROP_INFO = {
    "Rice":      {"season":"Kharif","duration":"90-120 days","water":"High",  "emoji":"🌾","hinglish":"Chawal"},
    "Wheat":     {"season":"Rabi",  "duration":"110-140 days","water":"Medium","emoji":"🌾","hinglish":"Gehu"},
    "Maize":     {"season":"Kharif","duration":"80-100 days","water":"Medium","emoji":"🌽","hinglish":"Makka"},
    "Cotton":    {"season":"Kharif","duration":"150-180 days","water":"Low",  "emoji":"🌿","hinglish":"Kapas"},
    "Sugarcane": {"season":"Annual","duration":"12-18 months","water":"High", "emoji":"🌿","hinglish":"Ganna"},
    "Chickpea":  {"season":"Rabi",  "duration":"90-120 days","water":"Low",   "emoji":"🫘","hinglish":"Chana"},
    "Lentil":    {"season":"Rabi",  "duration":"80-110 days","water":"Low",   "emoji":"🫘","hinglish":"Masoor"},
    "Mustard":   {"season":"Rabi",  "duration":"90-110 days","water":"Low",   "emoji":"🌼","hinglish":"Sarson"},
    "Sunflower": {"season":"Kharif","duration":"90-120 days","water":"Medium","emoji":"🌻","hinglish":"Surajmukhi"},
    "Potato":    {"season":"Rabi",  "duration":"70-100 days","water":"High",  "emoji":"🥔","hinglish":"Aloo"},
    "Tomato":    {"season":"Annual","duration":"60-90 days", "water":"Medium","emoji":"🍅","hinglish":"Tamatar"},
    "Onion":     {"season":"Rabi",  "duration":"100-130 days","water":"Medium","emoji":"🧅","hinglish":"Pyaaz"},
    "Mango":     {"season":"Summer","duration":"3-5 years",  "water":"Low",   "emoji":"🥭","hinglish":"Aam"},
    "Banana":    {"season":"Annual","duration":"10-12 months","water":"High", "emoji":"🍌","hinglish":"Kela"},
    "Soybean":   {"season":"Kharif","duration":"80-100 days","water":"Medium","emoji":"🫘","hinglish":"Soyabean"},
    "Groundnut": {"season":"Kharif","duration":"90-120 days","water":"Low",   "emoji":"🥜","hinglish":"Moongfali"},
    "Jute":      {"season":"Kharif","duration":"100-120 days","water":"High", "emoji":"🌿","hinglish":"Paat"},
    "Barley":    {"season":"Rabi",  "duration":"90-120 days","water":"Low",   "emoji":"🌾","hinglish":"Jau"},
    "Bajra":     {"season":"Kharif","duration":"60-90 days", "water":"Low",   "emoji":"🌾","hinglish":"Bajra"},
    "Jowar":     {"season":"Kharif","duration":"90-120 days","water":"Low",   "emoji":"🌾","hinglish":"Jwaar"},
}

# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        "message": "Kisan Mitra API",
        "crop_accuracy":    crop_meta['accuracy'],
        "disease_accuracy": dis_meta['accuracy'],
        "endpoints": ["/predict", "/detect-disease", "/all-crops", "/health"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "crop_model_accuracy":    crop_meta['accuracy'],
        "disease_model_accuracy": dis_meta['accuracy']
    })

# ── CROP PREDICTION ──────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feats = np.array([[
            float(data['N']),           float(data['P']),
            float(data['K']),           float(data['temperature']),
            float(data['humidity']),    float(data['ph']),
            float(data['rainfall'])
        ]])
        scaled = crop_scaler.transform(feats)
        probas = crop_model.predict_proba(scaled)[0]
        top3   = np.argsort(probas)[::-1][:3]

        results = []
        for idx in top3:
            name = crop_le.inverse_transform([idx])[0]
            results.append({
                "crop":       name,
                "hinglish":   CROP_INFO.get(name, {}).get("hinglish", name),
                "confidence": round(float(probas[idx]) * 100, 2),
                "emoji":      CROP_INFO.get(name, {}).get("emoji", "🌱"),
                "info":       CROP_INFO.get(name, {})
            })

        # Fertilizer advice
        N, P, K, ph = float(data['N']), float(data['P']), float(data['K']), float(data['ph'])
        fert = []
        fert.append({"nutrient":"N","status":"Low" if N<50 else "High" if N>120 else "Optimal",
                     "action": "Urea ya DAP 20-30 kg/acre daalein" if N<50 else "Nitrogen band karein" if N>120 else "Bilkul sahi ✓"})
        fert.append({"nutrient":"P","status":"Low" if P<35 else "Optimal",
                     "action":"SSP daalein" if P<35 else "Bilkul sahi ✓"})
        fert.append({"nutrient":"K","status":"Low" if K<40 else "Optimal",
                     "action":"MOP daalein" if K<40 else "Bilkul sahi ✓"})
        fert.append({"nutrient":"pH","status":"Acidic" if ph<6.0 else "Alkaline" if ph>7.8 else "Optimal",
                     "action":"Lime 2-3 quintal/acre daalein" if ph<6.0 else "Gypsum use karein" if ph>7.8 else "Bilkul sahi ✓"})

        return jsonify({"success": True, "top_recommendation": results[0],
                        "alternatives": results[1:], "fertilizer_advice": fert})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DISEASE DETECTION ────────────────────────
@app.route('/detect-disease', methods=['POST'])
def detect_disease():
    try:
        data = request.get_json()

        # Accept base64 image
        img_b64 = data.get('image', '')
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        # Decode image and extract color features
        img_bytes = base64.b64decode(img_b64.split(',')[-1])

        try:
            from PIL import Image
            img = Image.open(BytesIO(img_bytes)).convert('RGB').resize((100, 100))
            pixels = np.array(img).reshape(-1, 3).astype(float)

            r_mean = pixels[:, 0].mean()
            g_mean = pixels[:, 1].mean()
            b_mean = pixels[:, 2].mean()
            std    = pixels.std()

            # Brown pixel ratio (disease indicator)
            brown = np.sum((pixels[:,0] > 120) & (pixels[:,1] < 100) & (pixels[:,2] < 80)) / len(pixels)
            # Yellow pixel ratio
            yellow = np.sum((pixels[:,0] > 180) & (pixels[:,1] > 170) & (pixels[:,2] < 100)) / len(pixels)

        except Exception:
            # Fallback: random features for demo
            r_mean, g_mean, b_mean = 140.0, 130.0, 80.0
            std, brown, yellow = 35.0, 0.3, 0.2

        features = np.array([[
            r_mean, g_mean, b_mean,
            r_mean / (g_mean + 1),
            g_mean / (b_mean + 1),
            r_mean - g_mean,
            std,
            r_mean / (r_mean + g_mean + b_mean + 1),
            g_mean / (r_mean + g_mean + b_mean + 1),
            brown,
            yellow,
        ]])

        scaled = dis_scaler.transform(features)
        probas = dis_model.predict_proba(scaled)[0]
        top3   = np.argsort(probas)[::-1][:3]

        results = []
        for idx in top3:
            name    = dis_le.inverse_transform([idx])[0]
            remedy  = REMEDIES.get(str(name), {})
            results.append({
                "disease":    str(name),
                "confidence": round(float(probas[idx]) * 100, 1),
                "status":     remedy.get("status", "unknown"),
                "remedy":     remedy.get("remedy", ""),
                "prevention": remedy.get("prevention", ""),
            })

        return jsonify({
            "success":          True,
            "top_result":       results[0],
            "other_possibilities": results[1:],
            "color_analysis":   {
                "r_mean": round(r_mean, 1),
                "g_mean": round(g_mean, 1),
                "b_mean": round(b_mean, 1),
                "brown_ratio":  round(float(brown), 3),
                "yellow_ratio": round(float(yellow), 3),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/all-crops')
def all_crops():
    return jsonify({"crops": [{"name": k, **v} for k, v in CROP_INFO.items()], "total": len(CROP_INFO)})

@app.route('/crop-info/<name>')
def crop_info(name):
    n = name.capitalize()
    if n in CROP_INFO:
        return jsonify({"crop": n, "info": CROP_INFO[n]})
    return jsonify({"error": "Not found"}), 404

if __name__ == '__main__':
    print("🌱 Kisan Mitra API — http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
