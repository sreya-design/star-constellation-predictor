# Data

Place the dataset file here before running the notebook.

## Required File

**`constellations.csv`** — Star catalogue dataset

### Where to Download

The dataset was sourced from Hugging Face and compiles a catalogue of stars listed in Wikipedia.

1. Search for the star constellations dataset on [Hugging Face Datasets](https://huggingface.co/datasets)
2. Download `constellations.csv`
3. Place it in this `data/` directory

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| Name | string | Proper name of the star |
| Bayer | string | Bayer designation |
| Flamsteed | string | Flamsteed designation |
| Variable_Star | string | Variable star designation |
| HD_Number | float | Henry Draper Catalogue number |
| Hip_Number | float | Hipparcos Catalogue number |
| RA | string | Right Ascension (HMS format) |
| Dec | string | Declination (DMS format) |
| Vmag | float | Visual/Apparent Magnitude |
| AbsMag | float | Absolute Magnitude |
| Distance | float | Distance from Earth (parsecs) |
| SpectralClass | string | Spectral classification |
| Constellation | string | **Target variable** — constellation name |
| Notes | string | Additional notes |
| RA_deg | float | Right Ascension in decimal degrees |
| Dec_deg | float | Declination in decimal degrees |
| SpectralClass_Missing | int | Binary: 1 if spectral class is missing |

### Key Stats
- **Rows:** 11,681 stars
- **Target classes:** 88 constellations
- **Class distribution:** Uneven — Cygnus, Centaurus, Carina, Cassiopeia are most common
