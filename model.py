import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("📥 Loading dataset...")

# Load dataset
data = pd.read_csv("iris_dataset.csv")
print("✅ Dataset loaded")
print(data.head())

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("🌲 Training Random Forest model...")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "iris_model.pkl")
print("💾 Model saved as iris_model.pkl")
