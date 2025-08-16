import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("crop_samples.csv")

# Encode advisory labels
advisory_encoder = LabelEncoder()
df["advisory"] = advisory_encoder.fit_transform(df["advisory"])

# Features and target
X = df[["crop", "temperature_c", "humidity_pct"]]
y = df["advisory"]

# One-hot encode crop
X = pd.get_dummies(X, columns=["crop"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders
joblib.dump(model, "crop_advisory_model.pkl")
joblib.dump(advisory_encoder, "advisory_encoder.pkl")
joblib.dump(X.columns.tolist(), "crop_columns.pkl")
print("✅ Model + encoders saved.")

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=advisory_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=advisory_encoder.classes_,
            yticklabels=advisory_encoder.classes_)
plt.title("📉 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC AUC (if applicable)
try:
    num_classes = len(advisory_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
    y_score = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
    print(f"\n🎯 ROC AUC Score (OvR): {roc_auc:.2f}")
except ValueError as e:
    print(f"⚠️ ROC AUC skipped: {e}")
