"""
main.py
-------
Entry point: runs the full end-to-end pipeline —
EDA -> model training & tuning -> evaluation -> saved visuals & metrics.

Usage:
    python src/main.py
"""

from eda import run_eda
from train_model import run_training


def main():
    print("STEP 1: EXPLORATORY DATA ANALYSIS\n")
    run_eda()

    print("\n\nSTEP 2: MODEL TRAINING & EVALUATION\n")
    run_training()

    print("\n\nPipeline complete. Check the visuals/ and results/ folders for output.")


if __name__ == "__main__":
    main()
