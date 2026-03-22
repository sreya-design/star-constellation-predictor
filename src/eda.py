"""
eda.py
------
Generates exploratory data analysis plots for the star constellation dataset.
All plots are saved to results/.

Usage:
    python src/eda.py

Input:  data/constellations_cleaned.csv
Output: PNG plots in results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results", exist_ok=True)


def load_data(path="data/constellations_cleaned.csv"):
    df = pd.read_csv(path)
    df = df.replace("Unknown", pd.NA)
    return df


def plot_constellation_distribution(df):
    """Bar chart of top 20 constellations by star count."""
    plt.figure(figsize=(14, 5))
    df['Constellation'].value_counts().head(20).plot(kind='bar', color='steelblue', edgecolor='k')
    plt.title('Top 20 Constellations by Star Count')
    plt.xlabel('Constellation')
    plt.ylabel('Number of Stars')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/constellation_distribution.png", dpi=150)
    plt.close()
    print("Saved → results/constellation_distribution.png")


def plot_sky_distribution(df):
    """Scatter plot of stars on the celestial sphere (RA vs Dec)."""
    plt.figure(figsize=(12, 6))
    plt.scatter(df['RA_deg'], df['Dec_deg'], s=3, alpha=0.4, color='navy')
    plt.title('Sky Distribution of Stars (RA vs Dec)')
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.tight_layout()
    plt.savefig("results/sky_distribution.png", dpi=150)
    plt.close()
    print("Saved → results/sky_distribution.png")


def plot_magnitude_distributions(df):
    """Histograms for Vmag and AbsMag."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['Vmag'].dropna(), bins=40, edgecolor='k', color='skyblue')
    axes[0].set_title('Distribution of Visual Magnitude (Vmag)')
    axes[0].set_xlabel('Vmag')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(df['AbsMag'].dropna(), bins=50, edgecolor='k', color='salmon')
    axes[1].set_title('Distribution of Absolute Magnitude (AbsMag)')
    axes[1].set_xlabel('AbsMag')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("results/magnitude_distributions.png", dpi=150)
    plt.close()
    print("Saved → results/magnitude_distributions.png")


def plot_spectral_class_distribution(df):
    """Bar chart of spectral classes (first letter only)."""
    df_plot = df[df['Spectral_Class'].notna()].copy()
    df_plot['Spectral_Letter'] = df_plot['Spectral_Class'].astype(str).str[0]

    plt.figure(figsize=(10, 5))
    df_plot['Spectral_Letter'].value_counts().plot(kind='bar', color='mediumpurple', edgecolor='k')
    plt.title('Stars by Spectral Class (First Letter)')
    plt.xlabel('Spectral Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/spectral_class_distribution.png", dpi=150)
    plt.close()
    print("Saved → results/spectral_class_distribution.png")


def plot_correlation_matrix(df):
    """Heatmap of numeric feature correlations."""
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix (Numeric Features)')
    plt.tight_layout()
    plt.savefig("results/correlation_matrix.png", dpi=150)
    plt.close()
    print("Saved → results/correlation_matrix.png")


def plot_vmag_vs_distance(df):
    """Scatter plot of visual magnitude vs distance."""
    plt.figure(figsize=(9, 6))
    plt.scatter(df['Distance_ly'], df['Vmag'], s=4, alpha=0.4, color='teal')
    plt.title('Visual Magnitude vs Distance from Earth')
    plt.xlabel('Distance (light years)')
    plt.ylabel('Visual Magnitude (Vmag)')
    plt.tight_layout()
    plt.savefig("results/vmag_vs_distance.png", dpi=150)
    plt.close()
    print("Saved → results/vmag_vs_distance.png")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    plot_constellation_distribution(df)
    plot_sky_distribution(df)
    plot_magnitude_distributions(df)
    plot_spectral_class_distribution(df)
    plot_correlation_matrix(df)
    plot_vmag_vs_distance(df)

    print("\nAll EDA plots saved to results/")


if __name__ == "__main__":
    main()
