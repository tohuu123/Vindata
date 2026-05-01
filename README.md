# Datathon - Forecasting Model

This repository contains the training script and data for a daily revenue and COGS forecasting pipeline using XGBoost with seasonality memorization, seed ensembling, and macro-trend extrapolation.

## Overview

- Model: XGBoost regressors for `Revenue` and `COGS` trained to predict sales with hyperparameter tuning, seed ensembling, and macro-trend scaling for the submission period.
- Requirements: Python 3, a virtual environment, and packages listed in `requirements.txt`.

## Directory structure

Project root tree (top-level files and folders):

```text
.
├── train.py              # Main training and forecasting script
├── requirements.txt      # Python package dependencies
├── dataset/              # Input data files
│   ├── customers.csv
│   ├── geography.csv
│   ├── inventory.csv
│   ├── order_items.csv
│   ├── orders.csv
│   ├── payments.csv
│   ├── products.csv
│   ├── promotions.csv
│   ├── returns.csv
│   ├── reviews.csv
│   ├── sales.csv
│   ├── sample_submission.csv
│   ├── shipments.csv
│   └── web_traffic.csv
├── plots/                # Output directory for saved figures (created automatically)
├── submission.csv        # Output submission file 
└── README.md             # This file
```

## Environment setup

1. Create and activate a virtual environment: 

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you use Conda, create a Conda environment and install the same packages.

## Running the Model

To run the full pipeline (data loading, feature engineering, model training, and inference):

```bash
python train.py
```

This will:
1. Load data from the `dataset/` directory.
2. Perform feature engineering and calculate macro-trend multipliers.
3. Train XGBoost models with TimeSeriesSplit and Seed Ensembling.
4. Save feature importance plots to the `plots/` directory.
5. Generate the final predictions and save them to `submission.csv`.