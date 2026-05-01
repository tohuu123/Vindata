# Datathon - Forecasting Model

This repository contains the notebook and data for a daily revenue and COGS forecasting pipeline using XGBoost with recursive forecasting.

## Overview

- Model: XGBoost regressors for `Revenue` and `COGS` with recursive forecasting for the submission period.
- Requirements: Python 3, a virtual environment, and packages listed in `requirements.txt`.

## Directory structure

Project root tree (top-level files and folders):

```text
.
├── model.ipynb            # Main notebook 
├── requirements.txt      # Python package dependencies
├── dataset/              # Input data (sales.csv, sample_submission.csv, etc.)
│   ├── sales.csv
│   ├── sample_submission.csv
│   ├── customers.csv
│   └── ...
├── plots/                # Output directory for saved figures (created if missing)
├── submission.csv        # submission file 
└── README.md             # This file
```

## Environment setup

1. Create and activate a virtual environment: 

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you use Conda, create a Conda environment and install the same packages.

## Running `model.ipynb`

There are two common ways to run the notebook:

- Jupyter Notebook / JupyterLab:

```bash
jupyter notebook model.ipynb
```

  - Open the notebook in your browser and run cells sequentially.

- VS Code (Jupyter extension):

  - Open the workspace in VS Code and select the `.venv` Python interpreter.
  - Open `model.ipynb` and use the cell run controls or `Run All`.
