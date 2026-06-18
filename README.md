# 🌾 Kisan Mitra — Smart Crop Advisory System

> AI-powered crop recommendation + disease detection — Punjab ke kisan ke liye

## 🚀 Sirf EK Command Chalao

```bash
# 1. Dependencies install karo (sirf pehli baar)
pip install -r requirements.txt

# 2. Model train karo (sirf pehli baar)
python data/generate_dataset.py
python ml_models/train_model.py
python ml_models/train_disease_model.py

# 3. Server start karo
python backend/app.py

# 4. Browser mein kholo
# http://localhost:5000
```

**Bas! Ek server — frontend + backend dono ek saath.**

## 📁 Structure
```
smart-crop-advisory/
├── backend/
│   ├── app.py              ← Flask server (yahi chalao)
│   └── templates/
│       └── index.html      ← Frontend (Flask serve karta hai)
├── ml_models/
│   ├── train_model.py
│   ├── train_disease_model.py
│   └── saved_models/       ← Trained .pkl files
├── data/
│   ├── generate_dataset.py
│   └── crop_data.csv
└── requirements.txt
```

## 🎯 Features
- **Crop Recommendation** — 20 crops, 94.8% accuracy
- **Disease Detection** — 12 diseases, image upload
- **Fertilizer Advice** — NPK based
- **Crop Browser** — Season filter

## 👨‍💻 Developer
Sheran Asgar | B.Tech CSE, NIET Greater Noida
