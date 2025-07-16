import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Sample training data
data = {
    "crop": ["tomato", "rice", "wheat", "chilli", "sugarcane", "onion", "peanuts", "corn"] * 5,
    "temperature_c": [30, 25, 22, 35, 34, 28, 31, 29] * 5,
    "humidity_pct": [60, 70, 50, 80, 75, 65, 58, 62] * 5,
    "advisory": [
        "Water lightly", "Irrigate", "Reduce watering", "Spray pesticide", "Increase drainage",
        "Monitor for rot", "Mulch needed", "Check soil pH"
    ] * 5
}

df = pd.DataFrame(data)

# Encode crop and advisory
crop_encoder = LabelEncoder()
df["crop"] = crop_encoder.fit_transform(df["crop"])

advisory_encoder = LabelEncoder()
df["advisory"] = advisory_encoder.fit_transform(df["advisory"])

# Train model
X = df[["crop", "temperature_c", "humidity_pct"]]
y = df["advisory"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "crop_advisory_model.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(advisory_encoder, "advisory_encoder.pkl")
print("✅ Model and encoders saved.")
