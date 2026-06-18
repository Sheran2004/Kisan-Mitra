import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Windows + Linux dono pe kaam karega ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'crop_data.csv')

crop_conditions = {
    'Rice':      dict(N=(80,100), P=(40,60),  K=(40,60),  temp=(25,35), humidity=(80,90), ph=(6.0,7.0), rainfall=(200,250)),
    'Wheat':     dict(N=(100,120),P=(60,70),  K=(70,90),  temp=(10,20), humidity=(50,65), ph=(6.5,7.5), rainfall=(50,80)),
    'Maize':     dict(N=(90,110), P=(40,65),  K=(55,80),  temp=(22,28), humidity=(60,72), ph=(5.8,6.5), rainfall=(100,150)),
    'Cotton':    dict(N=(70,95),  P=(25,45),  K=(40,60),  temp=(30,38), humidity=(40,55), ph=(7.0,8.0), rainfall=(60,100)),
    'Sugarcane': dict(N=(110,140),P=(55,75),  K=(75,100), temp=(28,38), humidity=(70,85), ph=(6.5,7.5), rainfall=(150,200)),
    'Chickpea':  dict(N=(20,35),  P=(65,90),  K=(35,55),  temp=(15,22), humidity=(40,58), ph=(6.5,8.0), rainfall=(30,55)),
    'Lentil':    dict(N=(15,30),  P=(55,80),  K=(25,45),  temp=(12,20), humidity=(35,52), ph=(6.3,7.8), rainfall=(25,50)),
    'Mustard':   dict(N=(60,90),  P=(30,55),  K=(40,60),  temp=(10,18), humidity=(40,58), ph=(6.0,7.0), rainfall=(25,55)),
    'Sunflower': dict(N=(50,80),  P=(35,60),  K=(35,58),  temp=(22,28), humidity=(55,68), ph=(6.2,7.2), rainfall=(70,100)),
    'Potato':    dict(N=(100,130),P=(55,90),  K=(100,130),temp=(15,22), humidity=(70,85), ph=(5.2,6.2), rainfall=(80,120)),
    'Tomato':    dict(N=(75,105), P=(45,75),  K=(65,100), temp=(20,28), humidity=(60,78), ph=(6.0,6.8), rainfall=(65,110)),
    'Onion':     dict(N=(65,100), P=(38,62),  K=(55,85),  temp=(15,24), humidity=(52,72), ph=(6.2,7.2), rainfall=(55,90)),
    'Mango':     dict(N=(50,80),  P=(25,50),  K=(42,70),  temp=(28,38), humidity=(50,70), ph=(5.5,7.0), rainfall=(65,120)),
    'Banana':    dict(N=(100,140),P=(45,75),  K=(85,130), temp=(28,38), humidity=(75,90), ph=(5.5,6.8), rainfall=(140,220)),
    'Soybean':   dict(N=(20,40),  P=(55,85),  K=(45,72),  temp=(22,28), humidity=(58,72), ph=(6.2,7.2), rainfall=(90,140)),
    'Groundnut': dict(N=(20,45),  P=(42,72),  K=(38,62),  temp=(24,32), humidity=(52,72), ph=(5.8,6.8), rainfall=(75,120)),
    'Jute':      dict(N=(80,110), P=(32,58),  K=(42,68),  temp=(28,38), humidity=(80,90), ph=(6.2,7.2), rainfall=(170,250)),
    'Barley':    dict(N=(60,95),  P=(32,58),  K=(40,68),  temp=(8,18),  humidity=(45,65), ph=(6.2,7.8), rainfall=(30,70)),
    'Bajra':     dict(N=(50,85),  P=(25,50),  K=(30,55),  temp=(30,40), humidity=(40,58), ph=(6.2,8.0), rainfall=(30,70)),
    'Jowar':     dict(N=(52,85),  P=(27,52),  K=(32,58),  temp=(28,38), humidity=(42,62), ph=(6.3,8.0), rainfall=(38,85)),
}

records = []
per_crop = 500

for crop, cond in crop_conditions.items():
    for _ in range(per_crop):
        records.append({
            'N':           round(np.random.uniform(*cond['N']), 2),
            'P':           round(np.random.uniform(*cond['P']), 2),
            'K':           round(np.random.uniform(*cond['K']), 2),
            'temperature': round(np.random.uniform(*cond['temp']), 2),
            'humidity':    round(np.random.uniform(*cond['humidity']), 2),
            'ph':          round(np.random.uniform(*cond['ph']), 2),
            'rainfall':    round(np.random.uniform(*cond['rainfall']), 2),
            'label':       crop
        })

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Dataset saved: {OUTPUT_PATH}")
print(f"   Rows: {len(df)} | Crops: {df['label'].nunique()}")
