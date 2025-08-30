import os
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


# ── Config ──────────────────────────────
API_KEY = "a149dd4ae36548bb809135724251507"
KNN_MODEL_PATH = "crop_advisory_knn.pkl"
CROP_COLUMNS_PATH = "crop_columns.pkl"
DATA_PATH = "crop_samples_cleaned.csv"

# ── Flask ──────────────────────────────
app = Flask(__name__)
CORS(app)


# ── Load KNN model + data ──────────────
try:
    knn = joblib.load(KNN_MODEL_PATH)
    crop_columns = joblib.load(CROP_COLUMNS_PATH)
    df_data = pd.read_csv(DATA_PATH)
    print("✅ KNN model and data loaded.")
except Exception as e:
    knn, crop_columns, df_data = None, None, None
    print("❌ Error loading KNN model/data:", e)



# ── KNN-based advisory retrieval ──

def predict_detailed_advisory_knn(crop, season, soil_type, temp, humidity, rainfall):
    # Filter dataset for the selected crop only
    df_crop = df_data[df_data['crop'].str.lower() == crop.lower()].copy()
    if df_crop.empty:
        return [f"No advisory found for crop: {crop}"]
    # Prepare input as DataFrame with one-hot columns
    input_dict = {
        "crop": crop,
        "season": season,
        "soil_type": soil_type,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "rainfall_mm": rainfall
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    # Get columns for this crop subset
    crop_cols = [col for col in crop_columns if col in df_crop.columns or col in input_df.columns]
    for col in crop_cols:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[crop_cols]
    # One-hot encode crop subset
    X_crop = pd.get_dummies(df_crop[crop_cols])
    for col in crop_cols:
        if col not in X_crop:
            X_crop[col] = 0
    X_crop = X_crop[crop_cols]
    # Fit a temporary KNN on this crop's data
    from sklearn.neighbors import NearestNeighbors
    knn_crop = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn_crop.fit(X_crop)
    dist, idx = knn_crop.kneighbors(input_df, n_neighbors=1)
    nearest_row = df_crop.iloc[idx[0][0]]
    advisory = nearest_row["advisory_points"]
    points = [p.strip() for p in advisory.split('|')]
    return points

# ── Routes ─────────────────────────────
@app.route("/")
def home():
    return "✅ Agri-Weather AI backend is running."


@app.route("/weather")
def weather():
    city = request.args.get("city")
    crop = request.args.get("crop", "").lower()
    season = request.args.get("season", "Kharif")
    soil_type = request.args.get("soil_type", "alluvial")

    if not crop:
        return jsonify({"error": "Missing crop parameter."}), 400
    if not city:
        return jsonify({"error": "Missing location: provide ?city="}), 400

    q = city
    city_label = city

    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={q}&aqi=no"
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in data:
        return jsonify({"error": data["error"]["message"]}), 502

    cur = data["current"]
    temp, hum = cur["temp_c"], cur["humidity"]
    cond = cur["condition"]["text"]
    wind = cur.get("wind_kph", None)
    rainfall = cur.get("precip_mm", 0)

    advisory = predict_detailed_advisory_knn(crop, season, soil_type, temp, hum, rainfall)

    return jsonify({
        "city": city_label,
        "crop": crop,
        "season": season,
        "soil_type": soil_type,
        "temperature": temp,
        "humidity": hum,
        "rainfall": rainfall,
        "condition": cond,
        "wind": wind,
        "advisory": advisory,
        "error": None
    })

# ── Run ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
