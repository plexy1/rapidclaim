import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
df = pd.read_csv("backend/model/car_crash_data.csv")

# Remove leading/trailing spaces in headers
df.columns = df.columns.str.strip()

print("Cleaned CSV columns:", df.columns.tolist())

FEATURE_COLS = ['weather', 'time_of_day', 'road_type', 'speed']
TARGET_COL = 'severity'

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X = X.fillna(-1)

# Encode categoricals
for col in ['weather', 'time_of_day', 'road_type']:
    X[col] = X[col].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "backend/model/car_crash_model.joblib")
print("Model saved as backend/model/car_crash_model.joblib")
