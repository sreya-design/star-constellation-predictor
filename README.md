# 🌟 Star Constellation Predictor

A machine learning project that predicts which constellation a star belongs to based on its physical and positional properties.

## Research Question

> *"Given the properties of a star, such as coordinates, relative distance from Earth, and illuminance, can we predict the constellation it belongs to?"*

---

## Project Overview

Stars in the night sky are organized into 88 constellations, but identifying a star's constellation from raw data alone is a non-trivial classification problem. This project uses a catalogue of ~11,681 stars with known astronomical properties to train and evaluate several ML classifiers for multi-class constellation prediction.

---

## Dataset

- **Source:** [Hugging Face](https://huggingface.co/) — a star catalogue compiled from Wikipedia
- **Size:** 11,681 rows × 14–17 columns
- **Key features used:**
  | Feature | Description |
  |---|---|
  | `RA_deg` | Right Ascension (degrees) |
  | `Dec_deg` | Declination (degrees) |
  | `Vmag` | Visual/Apparent Magnitude |
  | `AbsMag` | Absolute Magnitude |
  | `Distance` | Distance from Earth (parsecs) |
  | `SpectralClass` | Spectral classification of the star |

> ⚠️ The raw dataset file (`constellations.csv`) is not included in this repository due to size. Download it from Hugging Face and place it in the `data/` folder.

---

## Methodology

### Data Cleaning
- Removed duplicate entries and rows with missing positional values (`RA_deg`, `Dec_deg`)
- Standardized Bayer designations and spectral class labels
- Created a binary indicator `SpectralClass_Missing` for rows with absent spectral data

### Exploratory Data Analysis
- Analyzed distribution of stars across constellations (uneven — Cygnus, Centaurus, Carina, and Cassiopeia are most common)
- Examined correlations: `Variable_Type` vs `SpectralClass` showed moderate correlation (~0.428)
- Investigated Simpson's Paradox potential in subgroups

### Dimensionality Reduction
Three techniques were compared:
- **PCA** — retained all features (95% variance threshold)
- **LDA** — reduced 6 features → 1 component (highest reduction)
- **UMAP** — non-linear reduction for visualization
- **Autoencoders** — deep learning-based compression

LDA was found most aggressive; however, the full 6-feature set was used for final modeling.

### Models Evaluated

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| k-NN (baseline) | — | — | k=7, Manhattan distance, distance-weighted |
| Random Forest | ~1.0 | ~0.65 | Macro F1: 0.69 |
| Gradient Boosting (XGBoost) | ~1.0 | ~0.67 | Slight improvement over RF |
| SVM (RBF kernel) | ~0.79 | ~0.63 | Less overfitting |
| **Tuned XGBoost** ✅ | — | **~0.70** | Best balance of performance & generalization |

### Final Model: Tuned XGBoost
- Hyperparameters optimized via stratified cross-validation
- Achieved ~70% test accuracy on a 88-class problem
- Best balance between overfitting and generalization

---

## Key Findings

- Constellation membership **can be predicted** with reasonable accuracy (~70%) using sky coordinates (`RA_deg`, `Dec_deg`) and brightness (`Vmag`, `AbsMag`)
- Positional features are the strongest predictors — constellations are fundamentally spatial groupings
- The multi-class nature (88 classes) and class imbalance make this a challenging problem
- XGBoost with tuned hyperparameters outperforms kNN and SVM

---

## Repository Structure

```
star-constellation-predictor/
│
├── notebooks/
│   └── star_constellation_predictor.ipynb   # Full analysis notebook
│
├── src/
│   └── (helper scripts, if any)
│
├── data/
│   └── (place constellations.csv here — not tracked by git)
│
├── results/
│   └── (output plots and metrics)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Download the dataset from Hugging Face and save it as `data/constellations.csv`
2. Open the notebook:
   ```bash
   jupyter notebook notebooks/star_constellation_predictor.ipynb
   ```
3. Run all cells sequentially

> **Note:** The notebook was originally developed on Google Colab. Replace `from google.colab import files` upload cells with `pd.read_csv("../data/constellations.csv")` for local execution.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-data--wrangling-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-lightblue)

- Python 3
- Pandas, NumPy, SciPy
- Scikit-learn (kNN, Random Forest, SVM, PCA, LDA)
- XGBoost
- UMAP
- Matplotlib, Seaborn

---

## Author

**Sreyana Gulapati**  
Academic Project — Data Science / Machine Learning  

---

## License

This project is for academic purposes.
