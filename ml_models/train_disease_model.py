import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 60)
print("  DISEASE DETECTION MODEL TRAINING")
print("=" * 60)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)

# Disease database with RGB color signatures + texture features
# Each disease has characteristic color patterns visible in images
DISEASES = {
    "Healthy":                {"r":(100,180), "g":(140,220), "b":(60,120),  "std":(10,30)},
    "Leaf Blight":            {"r":(160,220), "g":(120,170), "b":(50,90),   "std":(30,60)},
    "Powdery Mildew":         {"r":(210,240), "g":(205,235), "b":(195,225), "std":(5,20)},
    "Rust Disease":           {"r":(180,230), "g":(100,150), "b":(30,70),   "std":(25,50)},
    "Leaf Spot":              {"r":(140,190), "g":(90,130),  "b":(40,80),   "std":(35,65)},
    "Bacterial Wilt":         {"r":(170,210), "g":(155,190), "b":(100,140), "std":(20,45)},
    "Downy Mildew":           {"r":(150,200), "g":(170,210), "b":(130,170), "std":(15,35)},
    "Anthracnose":            {"r":(80,130),  "g":(60,100),  "b":(40,80),   "std":(40,70)},
    "Yellow Mosaic Virus":    {"r":(210,250), "g":(210,250), "b":(50,100),  "std":(10,30)},
    "Early Blight":           {"r":(160,200), "g":(130,170), "b":(50,90),   "std":(30,55)},
    "Late Blight":            {"r":(100,150), "g":(80,120),  "b":(60,100),  "std":(45,75)},
    "Nutrient Deficiency":    {"r":(200,240), "g":(190,230), "b":(100,140), "std":(15,35)},
}

REMEDIES = {
    "Healthy":             {"status":"healthy",  "remedy":"Fasal bilkul theek hai! Niyamit dekhbhal jaari rakho.", "prevention":"Regular monitoring karte raho."},
    "Leaf Blight":         {"status":"disease",  "remedy":"Mancozeb 2g/liter paani mein milakar spray karein. Beemari ke patte hata dein.", "prevention":"Zyada paani dene se bachein. Crop rotation karein."},
    "Powdery Mildew":      {"status":"disease",  "remedy":"Sulfur-based fungicide spray karein (Sulphex 3g/L). Subah spray best time hai.", "prevention":"Plants ke beech space rakhein. Overwatering avoid karein."},
    "Rust Disease":        {"status":"disease",  "remedy":"Propiconazole (Tilt 25EC) 1ml/liter paani mein spray karein.", "prevention":"Resistant varieties use karein. Infected plants hata dein."},
    "Leaf Spot":           {"status":"disease",  "remedy":"Copper Oxychloride 3g/liter spray karein. 2 hafte mein 2 baar.", "prevention":"Seedon ko overwater mat karein. Good drainage rakho."},
    "Bacterial Wilt":      {"status":"disease",  "remedy":"Streptomycin antibiotic spray (100ppm). Infected plants turant nikalo.", "prevention":"Certified disease-free seeds use karein."},
    "Downy Mildew":        {"status":"disease",  "remedy":"Metalaxyl + Mancozeb (Ridomil Gold) 2.5g/L spray karein.", "prevention":"Morning mein paani dein, raat ko nahi. Spacing badhao."},
    "Anthracnose":         {"status":"disease",  "remedy":"Carbendazim 1g/liter spray karein. Infected fruits/leaves hatao.", "prevention":"Healthy seeds use karein. Field saaf rakho."},
    "Yellow Mosaic Virus": {"status":"disease",  "remedy":"Koi direct ilaj nahi — infected plants hatao. Whitefly control karein (Imidacloprid spray).", "prevention":"Virus-resistant varieties lagao. Whitefly se bachao."},
    "Early Blight":        {"status":"disease",  "remedy":"Chlorothalonil 2g/liter spray karein. Weekly spraying karein.", "prevention":"Mulching karein. Infected patte hatate raho."},
    "Late Blight":         {"status":"disease",  "remedy":"Cymoxanil + Mancozeb spray karein. Turant action zaruri hai!", "prevention":"Certified seeds use karein. Avoid overhead irrigation."},
    "Nutrient Deficiency": {"status":"deficiency","remedy":"Soil test karwao. NPK balanced fertilizer daalein. Micronutrient spray karein.", "prevention":"Regular soil testing. Balanced fertilization plan banao."},
}

# Generate synthetic training data from color features
records = []
labels  = []
per_class = 400

for disease, color in DISEASES.items():
    for _ in range(per_class):
        r_mean = np.random.uniform(*color['r'])
        g_mean = np.random.uniform(*color['g'])
        b_mean = np.random.uniform(*color['b'])
        std    = np.random.uniform(*color['std'])

        records.append([
            r_mean, g_mean, b_mean,
            r_mean / (g_mean + 1),      # R/G ratio — key disease indicator
            g_mean / (b_mean + 1),      # G/B ratio
            (r_mean - g_mean),          # R-G difference
            std,                         # texture variance
            r_mean / (r_mean + g_mean + b_mean + 1),  # R ratio
            g_mean / (r_mean + g_mean + b_mean + 1),  # G ratio
            np.random.uniform(0.2, 0.9), # brown_ratio
            np.random.uniform(0.1, 0.8), # yellow_ratio
        ])
        labels.append(disease)

X = np.array(records)
y = np.array(labels)

le     = LabelEncoder()
scaler = StandardScaler()
y_enc  = le.fit_transform(y)
X_sc   = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"\nDisease Model Accuracy: {acc*100:.1f}%")

joblib.dump(clf,    os.path.join(MODEL_DIR, 'disease_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'disease_scaler.pkl'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'disease_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'disease_meta.json'), 'w') as f:
    json.dump({"accuracy": round(acc*100,1), "diseases": list(le.classes_), "remedies": REMEDIES}, f, indent=2)

print(f"Disease model saved! Classes: {list(le.classes_)}")
