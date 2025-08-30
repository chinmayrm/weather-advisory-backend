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
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not crop:
        return jsonify({"error": "Missing crop parameter."}), 400

    # If city is provided, geocode to lat/lon using Nominatim
    if city and not (lat and lon):
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            geo_headers = {"User-Agent": "AgriAI/1.0 (your@email.com)"}
            geo_resp = requests.get(geo_url, headers=geo_headers, timeout=10)
            geo_data = geo_resp.json()
            if not geo_data:
                return jsonify({"error": f"City '{city}' not found."}), 404
            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]
            city_label = geo_data[0]["display_name"]
        except Exception as e:
            return jsonify({"error": f"Geocoding error: {str(e)}"}), 500
    elif lat and lon:
        city_label = f"({lat},{lon})"
    else:
        return jsonify({"error": "Missing location: provide either ?city= or ?lat= and ?lon="}), 400

    # Now fetch weather from Open-Meteo
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relative_humidity_2m"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_data = weather_resp.json()
        if "current_weather" not in weather_data:
            return jsonify({"error": "Weather data not found for this location."}), 502
        temp = weather_data["current_weather"]["temperature"]
        wind = weather_data["current_weather"].get("windspeed", None)
        # Open-Meteo does not provide humidity in current_weather, so get from hourly
        humidity = None
        if "hourly" in weather_data and "relative_humidity_2m" in weather_data["hourly"]:
            # Find the closest hour to now
            import datetime
            now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            times = weather_data["hourly"]["time"]
            humidities = weather_data["hourly"]["relative_humidity_2m"]
            # Find index of closest time
            idx = 0
            for i, t in enumerate(times):
                tdt = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M")
                if tdt >= now:
                    idx = i
                    break
            humidity = humidities[idx]
        if humidity is None:
            humidity = 50  # fallback
        # Open-Meteo does not provide a text condition, so use weathercode
        code = weather_data["current_weather"].get("weathercode", 0)
        # Simple mapping for demo
        code_map = {
            0: "Clear",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Dense drizzle",
            56: "Freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Rain",
            65: "Heavy rain",
            66: "Freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow fall",
            73: "Snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail"
        }
        cond = code_map.get(code, "Unknown")
    except Exception as e:
        return jsonify({"error": f"Weather error: {str(e)}"}), 500

    advisory = get_ml_advisory(crop, temp, humidity)

    return jsonify({
        "city": city_label,
        "crop": crop,
        "temperature": temp,
        "humidity": humidity,
        "condition": cond,
        "wind": wind,
        "advisory": advisory,
        "error": None
    })

# ── Run ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
