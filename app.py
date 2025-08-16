import os
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ──────────────────────────────
API_KEY = "a149dd4ae36548bb809135724251507"
MODEL_PATH = "crop_advisory_model.pkl"
CROP_COLUMNS_PATH = "crop_columns.pkl"
ADVISORY_ENCODER_PATH = "advisory_encoder.pkl"

# ── Flask ──────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load model + encoders ──────────────
try:
    model = joblib.load(MODEL_PATH)
    crop_columns = joblib.load(CROP_COLUMNS_PATH)  # list of one-hot columns
    advisory_encoder = joblib.load(ADVISORY_ENCODER_PATH)
    print("✅ Model + encoders loaded.")
except Exception as e:
    model, crop_columns, advisory_encoder = None, None, None
    print("❌ Error loading model/encoders:", e)

# ── AI helper ──────────────────────────
def get_ml_advisory(crop, temp, humidity):
    if not model or not crop_columns or not advisory_encoder:
        return "⚠️ AI model or encoders not loaded."

    try:
        # Build input row
        input_dict = {col: 0 for col in crop_columns}
        input_dict["temperature_c"] = temp
        input_dict["humidity_pct"] = humidity

        crop_col = f"crop_{crop}"
        if crop_col not in input_dict:
            return f"⚠️ Crop '{crop}' not recognized."
        input_dict[crop_col] = 1

        # Create dataframe & align with model
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model.feature_names_in_]

        # Predict
        pred_encoded = model.predict(input_df)[0]
        advisory = advisory_encoder.inverse_transform([pred_encoded])[0]
        return advisory

    except Exception as e:
        print("ML error:", e)
        return "⚠️ AI error."

# ── Routes ─────────────────────────────
@app.route("/")
def home():
    return "✅ Agri-Weather AI backend is running."

@app.route("/weather")
def weather():
    city = request.args.get("city")
    crop = request.args.get("crop", "").lower()

    if not city:
        return jsonify({"error": "Missing ?city="}), 400

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

# ── Run ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
