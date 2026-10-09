# Prediction of Agriculture Crop Production in India

## About
A beginner-friendly Python Machine Learning project that demonstrates how historical agricultural records can be used to estimate crop production.

## Important data note
`crop_production_sample.csv` is a small **illustrative sample dataset created for learning**. It is not official government data and must not be presented as real agricultural statistics. The small dataset is only for testing that the code runs. Replace it with a verified dataset before drawing conclusions or claiming meaningful model accuracy.

## Features
- Reads agriculture records from CSV
- Checks required columns and removes incomplete rows
- Converts categorical columns using OneHotEncoder
- Trains a Random Forest regression model
- Reports MAE, RMSE and R² on a held-out split
- Accepts user inputs and estimates production

## Required columns
- `State`
- `District`
- `Crop`
- `Season`
- `Area_hectares`
- `Production_tonnes` (target column)

## Run on Windows
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
2. During installation, tick **Add python.exe to PATH**.
3. Extract this ZIP folder.
4. Open the folder in VS Code.
5. Open Terminal > New Terminal.
6. Run:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install -r requirements.txt
   py main.py
   ```
   If `py` is not recognized, try `python` instead after installing Python.
7. Enter State, District, Crop, Season and cultivated area when prompted.

## Files
- `main.py` — training, evaluation and interactive prediction
- `crop_production_sample.csv` — illustrative sample data
- `requirements.txt` — required Python libraries

## Suggested real dataset
Find a verified agricultural production dataset from an official government source or a reputable dataset provider (for example, search data.gov.in or Kaggle). Check its source, units, time period, and column definitions before using it. You may need to rename columns to match the required names.

## Limitations
This is a starter implementation, not the original code from the internship. A reliable crop-production model needs a sufficiently large, verified dataset, proper time-aware validation, and additional relevant variables such as year, rainfall, soil and weather where available.
