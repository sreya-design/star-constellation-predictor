# Source Code

| File | Purpose |
|------|---------|
| `preprocess.py` | Cleans raw `constellations.csv` → saves `constellations_cleaned.csv` |
| `eda.py` | Generates exploratory data analysis plots → saved to `results/` |
| `train.py` | Trains and evaluates all four ML models (kNN, RF, XGBoost, Tuned XGBoost) |

## Run Order

```bash
# 1. Clean the data
python src/preprocess.py

# 2. Generate EDA plots
python src/eda.py

# 3. Train and evaluate all models
python src/train.py
```
