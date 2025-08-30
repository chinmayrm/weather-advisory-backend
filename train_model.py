import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.neighbors import NearestNeighbors



# Load dataset and filter out any bad header rows or non-numeric data
df = pd.read_csv("crop_samples.csv")
df = df[pd.to_numeric(df["temperature_c"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["humidity_pct"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["rainfall_mm"], errors="coerce").notnull()]
df["temperature_c"] = df["temperature_c"].astype(float)
df["humidity_pct"] = df["humidity_pct"].astype(float)
df["rainfall_mm"] = df["rainfall_mm"].astype(float)
df["advisory_points"] = df["advisory_points"].astype(str)

# Encode target
advisory_encoder = LabelEncoder()
df["advisory_points_encoded"] = advisory_encoder.fit_transform(df["advisory_points"])

# Features and target
X = df[["crop", "season", "soil_type", "temperature_c", "humidity_pct", "rainfall_mm"]]
y = df["advisory_points_encoded"]

# One-hot encode categorical features
X = pd.get_dummies(X, columns=["crop", "season", "soil_type"])

# Split data (no stratify due to small class sizes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train KNN on all data (retrieval-based)
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
knn.fit(X, y=None)

# Save KNN model, feature columns, and the full dataframe for retrieval
joblib.dump(knn, "crop_advisory_knn.pkl")
joblib.dump(X.columns.tolist(), "crop_columns.pkl")
df.to_csv("crop_samples_cleaned.csv", index=False)
print("✅ KNN model and data saved for advisory retrieval.")
