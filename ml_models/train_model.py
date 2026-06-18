import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("  SMART CROP ADVISORY - ML MODEL TRAINING")
print("=" * 60)

# ── Paths (Windows + Linux dono pe kaam karega) ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_PATH  = os.path.join(ROOT_DIR, 'data', 'crop_data.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Data ──
df = pd.read_csv(DATA_PATH)
print(f"\n[1] Data loaded: {df.shape[0]} rows, {df['label'].nunique()} crops")

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features].values
y = df['label'].values

# ── Encode + Scale ──
le     = LabelEncoder()
scaler = StandardScaler()
y_enc  = le.fit_transform(y)
X_sc   = scaler.fit_transform(X)

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
print(f"[2] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train Model ──
print("[3] Training Random Forest (300 trees)...")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
print(f"[4] Accuracy: {acc*100:.2f}%")

# ── Save ──
joblib.dump(rf,     os.path.join(MODEL_DIR, 'crop_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'label_encoder.pkl'))

meta = {
    "accuracy": round(acc*100, 2),
    "features": features,
    "classes":  list(le.classes_),
    "n_estimators": 300
}
with open(os.path.join(MODEL_DIR, 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[5] Models saved to: {MODEL_DIR}")

# ── Quick Test ──
print("\n[6] Quick Test:")
tests = [
    ("Rice",   [90, 50, 50, 30, 85, 6.5, 220]),
    ("Wheat",  [110,65, 80, 15, 57, 7.0, 65]),
    ("Potato", [115,72,115, 18, 77, 5.8, 100]),
]
for expected, vals in tests:
    pred = le.inverse_transform(rf.predict(scaler.transform([vals])))[0]
    conf = rf.predict_proba(scaler.transform([vals]))[0].max() * 100
    mark = "✓" if pred == expected else "✗"
    print(f"  {mark} Expected: {expected:10s} → Got: {pred:10s} ({conf:.1f}%)")

print(f"\n{'='*60}")
print(f"  DONE! Accuracy: {acc*100:.2f}%")
print(f"{'='*60}")
