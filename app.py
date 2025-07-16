import os, requests, joblib, pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

API_KEY = "a149dd4ae36548bb809135724251507"
MODEL_PATH = "crop_advisory_model.pkl"
ENCODER_PATH = "crop_encoder.pkl"
ADVICE_ENCODER_PATH = "advisory_encoder.pkl"

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load(MODEL_PATH)
    crop_encoder = joblib.load(ENCODER_PATH)
    advisory_encoder = joblib.load(ADVICE_ENCODER_PATH)
    print("✅ Model & encoders loaded.")
except Exception as e:
    print("❌ Error loading model/encoders:", e)
    model = crop_encoder = advisory_encoder = None

def get_ml_advisory(crop, temp, humidity):
    try:
        crop_encoded = crop_encoder.transform([crop])[0]
        input_df = pd.DataFrame([[crop_encoded, temp, humidity]],
                                columns=["crop", "temperature_c", "humidity_pct"])
        pred_encoded = model.predict(input_df)[0]
        return advisory_encoder.inverse_transform([pred_encoded])[0]
    except Exception as e:
        print("ML error:", e)
        return "⚠️ AI error."

@app.route("/")
def home():
    return "✅ Agri‑Weather AI backend running."

@app.route("/weather")
def weather():
    city = request.args.get("city")
    crop = request.args.get("crop", "").lower()

    if not city:
        return jsonify({"error":"Missing ?city="}),400

    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        data = requests.get(url,timeout=10).json()
    except Exception as e:
        return jsonify({"error":str(e)}),500

    if "error" in data:
        return jsonify({"error":data["error"]["message"]}),502

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=True)
