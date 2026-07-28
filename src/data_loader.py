"""
data_loader.py
---------------
Utility functions to load and lightly validate the heart disease patient dataset.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease.csv"

COLUMN_DESCRIPTIONS = {
    "age": "Age of the patient in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment",
    "ca": "Number of major vessels (0-4) colored by fluoroscopy",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
    "target": "Presence of heart disease (1 = disease, 0 = no disease)",
}


def load_data(path: Path = DATA_PATH, clean: bool = True) -> pd.DataFrame:
    """Load the heart disease dataset from CSV into a pandas DataFrame.

    If clean=True (default), exact duplicate rows are removed as a basic
    data-cleaning step before analysis/modeling.
    """
    df = pd.read_csv(path)
    if clean:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"[data_loader] Removed {removed} duplicate row(s). New shape: {df.shape}")
    return df


def basic_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset: shape, dtypes, missing values."""
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())


if __name__ == "__main__":
    data = load_data()
    basic_summary(data)
