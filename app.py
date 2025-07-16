import os
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Configuration ──────────────────────────────────────────────
API_KEY = "a149dd4ae36548bb809135724251507"  # WeatherAPI key
MODEL_PATH = "crop_advisory_model.pkl"
CROP_ENCODER_PATH = "crop_encoder.pkl"
ADVICE_ENCODER_PATH = "advice_encoder.pkl"

# ── Flask setup ────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load model and encoders ────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    crop_encoder = joblib.load(CROP_ENCODER_PATH)
    advice_encoder = joblib.load(ADVICE_ENCODER_PATH)
    print("✅ Model and encoders loaded.")
except Exception as e:
    model = None
    crop_encoder = None
    advice_encoder = None
    print("❌ Error loading model/encoders:", e)

# ── Helper: AI prediction ───────────────────────────────────────
def get_ml_advisory(crop, temp, humidity):
    if not model or not crop_encoder or not advice_encoder:
        return "⚠️ AI model or encoders not loaded."

    try:
        if crop not in crop_encoder.classes_:
            return f"⚠️ Crop '{crop}' not recognized."

        # Create zero vector for all crops
        crop_encoded = crop_encoder.transform([crop])[0]
        crop_ohe = [1 if i == crop_encoded else 0 for i in range(len(crop_encoder.classes_))]

        # Build full input feature vector
        input_data = crop_ohe + [temp, humidity]

        # Create all feature names
        feature_names = [f"crop_{c}" for c in crop_encoder.classes_] + ["temperature_c", "humidity_pct"]
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Ensure order and completeness
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0  # fill missing expected features
        input_df = input_df[model_features]

        # Predict
        pred_encoded = model.predict(input_df)[0]
        advisory = advice_encoder.inverse_transform([pred_encoded])[0]
        return advisory

    except Exception as e:
        print("ML error:", e)
        return "⚠️ AI error."



# ── Routes ──────────────────────────────────────────────────────
@app.route("/")
def home():
    return "✅ Agri‑Weather AI backend is running."

@app.route("/weather")
def weather():
    city = request.args.get("city")
    crop = request.args.get("crop", "").lower()

    if not city:
        return jsonify({"error": "Missing ?city="}), 400

    # Fetch weather
    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in data:
        return jsonify({"error": data["error"]["message"]}), 502

    cur = data["current"]
    temp, hum = cur["temp_c"], cur["humidity"]
    cond = cur["condition"]["text"]

    advisory = get_ml_advisory(crop, temp, hum)

    return jsonify({
        "city": city,
        "crop": crop,
        "temperature": temp,
        "humidity": hum,
        "condition": cond,
        "advisory": advisory,
        "error": None
    })

# ── Run locally ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
