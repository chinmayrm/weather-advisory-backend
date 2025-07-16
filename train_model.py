import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("crop_samples.csv")

# Encode target labels
label_encoder = LabelEncoder()
df["advisory"] = label_encoder.fit_transform(df["advisory"])

# Split features and target
X = df[["crop", "temperature_c", "humidity_pct"]]
y = df["advisory"]

# Encode 'crop' using one-hot
X = pd.get_dummies(X, columns=["crop"])

# Dynamically safe test size (at least 1 sample per class)
num_classes = len(np.unique(y))
test_size = max(0.2, num_classes / len(df))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crop_advisory_model.pkl")
print("✅ Model trained and saved.")

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("📉 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC AUC score (only if enough class variety in test set)
try:
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
    y_score = model.predict_proba(X_test)

    roc_auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
    print(f"\n🎯 ROC AUC Score (OvR): {roc_auc:.2f}")
except ValueError as e:
    print(f"⚠️ ROC AUC skipped: {e}")
