from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)  # ✅ This allows cross-origin requests (fixes GitHub Pages issue)


# ── 1. CONFIG ─────────────────────────────────────────────
API_KEY = "a149dd4ae36548bb809135724251507"  # ← your WeatherAPI.com key

# ── 2. ROOT ROUTE  (fixes Render "Application Loading") ──
@app.route("/")
def home():
    return "✅ Weather Advisory Backend is running!"

# ── 3. /weather  MAIN API ROUTE ───────────────────────────
@app.route("/weather")
def weather():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "Query param ?city= is required"}), 400

    # 3‑a. Call WeatherAPI.com
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 3‑b. Handle API errors
    if "error" in data:
        return jsonify({"error": data["error"]["message"]}), 502

    # 3‑c. Extract useful fields
    cur = data["current"]
    temp = cur["temp_c"]
    humidity = cur["humidity"]
    condition = cur["condition"]["text"]

    # 3‑d. Very simple advisory logic
    if temp > 35:
        advisory = "🔥 Very hot. Irrigate crops early morning."
    elif humidity > 80:
        advisory = "💧 High humidity. Avoid pesticide spraying."
    else:
        advisory = "✅ Conditions normal. Proceed with regular activity."

    return jsonify(
        {
            "city": city,
            "temperature": temp,
            "humidity": humidity,
            "condition": condition,
            "advisory": advisory,
            "error": None,
        }
    )

# ── 4. RUN ────────────────────────────────────────────────
if __name__ == "__main__":
    # Render sets PORT env‑var; default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
