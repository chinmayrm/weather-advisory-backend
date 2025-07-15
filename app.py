from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

API_KEY = "a149dd4ae36548bb809135724251507"  # WeatherAPI key

@app.route("/")
def home():
    return "✅ Agri-Weather Advisory Backend is Live"

@app.route("/weather")
def weather():
    city = request.args.get("city")
    crop = request.args.get("crop", "").lower()

    if not city:
        return jsonify({"error": "Missing ?city= parameter"}), 400

    # Fetch weather
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in data:
        return jsonify({"error": data["error"]["message"]}), 502

    current = data["current"]
    temp = current["temp_c"]
    humidity = current["humidity"]
    condition = current["condition"]["text"]

    # Smart Crop-Specific Advisory
    advisory = get_crop_advisory(temp, humidity, condition, crop)

    return jsonify({
        "city": city,
        "crop": crop,
        "temperature": temp,
        "humidity": humidity,
        "condition": condition,
        "advisory": advisory,
        "error": None
    })


def get_crop_advisory(temp, humidity, condition, crop):
    if crop == "cotton":
        if humidity > 80:
            return "🛑 Avoid pesticide spraying today due to high humidity. Risk of wash-off."
        elif temp > 35:
            return "🔥 Very hot. Irrigate cotton fields early morning or late evening."
        else:
            return "✅ Suitable weather for cotton. Monitor pest activity."

    elif crop == "paddy":
        if "rain" in condition.lower():
            return "🌧️ Rain expected. Delay nitrogen fertilizer application."
        elif humidity > 85:
            return "💧 High humidity. Monitor for blast disease in paddy."
        else:
            return "✅ Good weather for paddy growth."

    elif crop == "tomato":
        if temp < 20:
            return "❄️ Low temperature may affect fruiting. Protect young plants."
        elif humidity > 85:
            return "🦠 Risk of fungal infection. Monitor leaves closely."
        else:
            return "✅ Favorable for tomato farming."

    else:
        return "✅ General advisory: Weather looks normal. Proceed with usual farm tasks."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
