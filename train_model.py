"""
train_model.py
Creates a synthetic crop‑weather dataset, trains a Random‑Forest pipeline,
saves both the fitted model (including preprocessing) as crop_advisory_model.pkl
"""

import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ───────────────────────────────────────────────────────────────
# 1) Generate synthetic dataset (300 rows, 3 crops)
np.random.seed(42)
def rule(crop, t, h):
    if crop == "cotton":
        if h > 80:            return "🛑 Avoid pesticide spraying (cotton, high humidity)."
        if t > 37:            return "🔥 Heat stress—irrigate cotton early morning."
        return "✅ Cotton conditions normal."
    if crop == "paddy":
        if h > 88:            return "💧 High humidity—watch blast disease in paddy."
        if t < 25:            return "🌧️ Rain likely—delay nitrogen fertilizer."
        return "✅ Paddy conditions normal."
    if crop == "tomato":
        if h > 85:            return "🦠 Fungal risk in tomato—ensure ventilation."
        if t < 20:            return "❄️ Protect tomato seedlings from cold."
        return "✅ Tomato conditions normal."
    return "✅ General advisory."

rows=[]
for _ in range(300):
    crop = np.random.choice(["cotton","paddy","tomato"])
    temp = round(np.random.uniform(18,42),1)
    hum  = np.random.randint(60,100)
    rows.append([crop,temp,hum,rule(crop,temp,hum)])

df = pd.DataFrame(rows, columns=["crop","temperature_c","humidity_pct","advisory"])
df.to_csv("crop_weather_dataset.csv",index=False)
print("✅ Synthetic dataset saved → crop_weather_dataset.csv")

# ───────────────────────────────────────────────────────────────
# 2) Build preprocessing + model pipeline
X = df.drop("advisory",axis=1)
y = df["advisory"]

prep = ColumnTransformer([
    ("num", StandardScaler(), ["temperature_c","humidity_pct"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["crop"])
])

model = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)

pipe = Pipeline(steps=[("prep",prep),("model",model)])

# ───────────────────────────────────────────────────────────────
# 3) Train + quick validation printout
X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
pipe.fit(X_train,y_train)
print("── Validation report ──")
print(classification_report(y_val, pipe.predict(X_val)))

# ───────────────────────────────────────────────────────────────
# 4) Save fitted pipeline
joblib.dump(pipe,"crop_advisory_model.pkl")
print("✅ Model saved → crop_advisory_model.pkl")
