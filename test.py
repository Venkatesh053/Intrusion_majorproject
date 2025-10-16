import joblib
import pandas as pd

# Load your processed training dataset again
df = pd.read_csv("./datasets/nslkdd/nslkdd_processed.csv")

# Drop label column
X = df.drop("labels", axis=1)

# Get all columns (after one-hot encoding)
categorical_cols = ["protocol_type", "service", "flag"]
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Save the feature list
joblib.dump(X_encoded.columns.tolist(), "./models/model_features.pkl")
print("✅ Saved model_features.pkl with", len(X_encoded.columns), "features.")
