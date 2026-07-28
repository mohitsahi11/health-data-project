"""
eda.py
------
Exploratory Data Analysis for the heart disease patient dataset.
Generates and saves visualizations to the visuals/ directory, and prints
key statistics to the console / results log.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from data_loader import load_data, basic_summary

sns.set_theme(style="whitegrid")
VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)


def plot_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x="target", hue="target", data=df,
                        palette={0: "#e74c3c", 1: "#2ecc71"}, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
    plt.title("Distribution of Heart Disease Diagnosis")
    plt.xlabel("")
    plt.ylabel("Number of Patients")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "target_distribution.png", dpi=150)
    plt.close()


def plot_age_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="age", hue="target", kde=True, bins=20,
                 palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.6)
    plt.title("Age Distribution by Heart Disease Diagnosis")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Patients")
    plt.legend(title="Diagnosis", labels=["Disease", "No Disease"])
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "age_distribution.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap of Patient Features")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_chest_pain_vs_target(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="cp", hue="target", data=df, palette={0: "#2ecc71", 1: "#e74c3c"})
    plt.title("Chest Pain Type vs Heart Disease")
    plt.xlabel("Chest Pain Type (0-3)")
    plt.ylabel("Number of Patients")
    plt.legend(title="Diagnosis", labels=["No Disease", "Disease"])
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "chest_pain_vs_target.png", dpi=150)
    plt.close()


def plot_age_vs_maxheartrate(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="age", y="thalach", hue="target", data=df,
                     palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.8)
    plt.title("Age vs Maximum Heart Rate Achieved")
    plt.xlabel("Age (years)")
    plt.ylabel("Max Heart Rate Achieved")
    plt.legend(title="Diagnosis", labels=["No Disease", "Disease"])
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "age_vs_maxheartrate.png", dpi=150)
    plt.close()


def run_eda() -> pd.DataFrame:
    df = load_data()
    print("=" * 60)
    print("BASIC DATASET SUMMARY")
    print("=" * 60)
    basic_summary(df)

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe().T)

    print("\nGenerating visualizations...")
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    plot_chest_pain_vs_target(df)
    plot_age_vs_maxheartrate(df)
    print(f"Saved 5 charts to: {VISUALS_DIR}")

    return df


if __name__ == "__main__":
    run_eda()
