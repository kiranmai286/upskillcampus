"""
Prediction of Agriculture Crop Production in India
Educational starter project.

IMPORTANT:
- crop_production_sample.csv contains illustrative sample data, not official statistics.
- For meaningful results, replace it with a verified real agricultural dataset.
- The model estimates production in tonnes from the features available in the CSV.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path(__file__).with_name("crop_production_sample.csv")
TARGET = "Production_tonnes"
CATEGORICAL = ["State", "District", "Crop", "Season"]
NUMERIC = ["Area_hectares"]

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Keep the CSV in the same folder as this script."
        )

    data = pd.read_csv(DATA_FILE)
    required = CATEGORICAL + NUMERIC + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].dropna()
    if len(data) < 10:
        raise ValueError("Please provide at least 10 complete rows to train the demonstration model.")

    X = data[CATEGORICAL + NUMERIC]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", "passthrough", NUMERIC),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Model Evaluation (sample dataset only) ===")
    print(f"Rows used: {len(data)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f} tonnes")
    print(f"Root Mean Squared Error (RMSE): {mean_squared_error(y_test, predictions) ** 0.5:.2f} tonnes")
    print(f"R² score: {r2_score(y_test, predictions):.3f}")
    print("\nReminder: these metrics are not evidence of real-world accuracy because this is a tiny illustrative dataset.")

    print("\n=== Try a prediction ===")
    print("Enter values matching a row type in your dataset.")
    state = input("State (e.g., Andhra Pradesh): ").strip()
    district = input("District (e.g., Guntur): ").strip()
    crop = input("Crop (e.g., Rice): ").strip()
    season = input("Season (e.g., Kharif): ").strip()
    try:
        area = float(input("Cultivated area in hectares (e.g., 120): "))
        if area <= 0:
            raise ValueError
    except ValueError:
        print("Area must be a positive number.")
        return

    example = pd.DataFrame([{
        "State": state,
        "District": district,
        "Crop": crop,
        "Season": season,
        "Area_hectares": area,
    }])
    estimate = model.predict(example)[0]
    print(f"\nEstimated crop production: {estimate:.2f} tonnes")
    print("This is a learning demonstration, not an official agricultural forecast.")

if __name__ == "__main__":
    main()
