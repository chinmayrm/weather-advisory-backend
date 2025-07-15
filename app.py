from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_KEY = "a149dd4ae36548bb809135724251507"

def get_weather(city):
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return {"error": data["error"]["message"]}

    current = data["current"]
    return {
        "temperature": current["temp_c"],
        "humidity": current["humidity"],
        "condition": current["condition"]["text"],
        "error": None
    }

def make_advisory(temp, humidity):
    if temp is None:
        return "Weather data unavailable."
    if temp > 35:
        return "🔥 Very hot. Irrigate crops early morning."
    if humidity and humidity > 80:
        return "💧 High humidity. Avoid pesticide spraying."
    return "✅ Conditions normal. Proceed with regular activity."

@app.route("/")
def home():
    return "✅ Weather Advisory API is running!"

@app.route("/weather")
def weather_api():
    city = request.args.get("city", default="Gadag")
    weather = get_weather(city)
    advisory = make_advisory(weather.get("temperature"), weather.get("humidity"))

    return jsonify({
        "city": city,
        "temperature": weather.get("temperature"),
        "humidity": weather.get("humidity"),
        "condition": weather.get("condition", "N/A"),
        "advisory": advisory,
        "error": weather.get("error")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
