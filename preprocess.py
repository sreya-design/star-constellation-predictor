"""
preprocess.py
-------------
Cleans the raw constellations.csv dataset and saves a cleaned version.

Usage:
    python src/preprocess.py

Input:  data/constellations.csv
Output: data/constellations_cleaned.csv
"""

import pandas as pd
import numpy as np
import re


def hms_to_deg(hms_str):
    """Convert Right Ascension from HMS string to decimal degrees."""
    try:
        if pd.isna(hms_str):
            return np.nan
        hms_str = str(hms_str).replace('h', ' ').replace('m', ' ').replace('s', ' ')
        parts = re.split('[ :]+', hms_str.strip())
        parts = [p for p in parts if p]
        if not parts or not parts[0].replace('.', '', 1).isdigit():
            return np.nan
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit() else 0
        s = float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '', 1).isdigit() else 0
        return h * 15 + m * 0.25 + s * 0.0041667
    except Exception:
        return np.nan


def dms_to_deg(dms_str):
    """Convert Declination from DMS string to decimal degrees."""
    try:
        if pd.isna(dms_str):
            return np.nan
        dms_str = (str(dms_str)
                   .replace('−', '-').replace('°', ' ')
                   .replace('′', ' ').replace("'", ' ')
                   .replace('"', ' ').replace('″', ' '))
        parts = re.split('[ :]+', dms_str.strip())
        parts = [p for p in parts if p]
        if not parts or not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            return np.nan
        sign = -1 if '-' in parts[0] else 1
        d = abs(float(parts[0]))
        m = float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit() else 0
        s = float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '', 1).isdigit() else 0
        return sign * (d + m / 60 + s / 3600)
    except Exception:
        return np.nan


def preprocess(input_path="data/constellations.csv",
               output_path="data/constellations_cleaned.csv"):
    print(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path)

    # Rename columns for clarity
    data = data.rename(columns={
        'Right Ascension': 'RA_hms',
        'Declination': 'Dec_dms',
        'Visual Magnitude': 'Vmag',
        'Absolute Magnitude': 'AbsMag',
        'Distance': 'Distance_ly',
        'Spectral class': 'Spectral_Class',
        'Bayer': 'Bayer_Designation',
        'Flamsteed': 'Flamsteed_Number',
        'Variable Star': 'Variable_Type',
        'Henry Draper Number': 'HD_Number',
        'Hipparcos Number': 'Hip_Number'
    })

    # Convert RA/Dec to decimal degrees
    data['RA_deg'] = data['RA_hms'].apply(hms_to_deg)
    data['Dec_deg'] = data['Dec_dms'].apply(dms_to_deg)

    # Convert numeric columns
    for col in ['Distance_ly', 'Vmag', 'AbsMag']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Impute missing numeric values with column median
    for col in ['RA_deg', 'Dec_deg', 'Distance_ly', 'Vmag', 'AbsMag']:
        data[col] = data[col].fillna(data[col].median())

    # Drop rows where both RA and Dec are missing (edge case)
    data = data.dropna(subset=['RA_deg', 'Dec_deg'], how='all')

    # Remove duplicates
    data = data.drop_duplicates()

    # Remove extreme outliers
    data = data[(data['AbsMag'] <= 1000) & (data['Distance_ly'] <= 1e8)]

    # Fill categorical columns with 'Unknown'
    categorical_cols = ['Spectral_Class', 'Bayer_Designation', 'Flamsteed_Number',
                        'Variable_Type', 'HD_Number', 'Hip_Number']
    for col in categorical_cols:
        data[col] = data[col].fillna('Unknown')

    # Flag rows where spectral class is missing
    data['SpectralClass_Missing'] = (data['Spectral_Class'] == 'Unknown').astype(int)

    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Shape: {data.shape}")
    return data


if __name__ == "__main__":
    preprocess()
