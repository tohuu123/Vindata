import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import random
import itertools
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

# --- 1. CONFIGURATION ---
DATA_DIR = "dataset"
SALES_PATH = os.path.join(DATA_DIR, "sales.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_SUB_PATH = "submission.csv"
PLOT_DIR = "plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- 2. DATA LOADING & MACRO TREND CALCULATION ---
print("Loading data...")
full_sales_df = pd.read_csv(SALES_PATH)
full_sales_df['Date'] = pd.to_datetime(full_sales_df['Date'])
full_sales_df = full_sales_df.sort_values('Date').reset_index(drop=True)

# Tính toán mức tăng trưởng vĩ mô (Macro-Trend) từ 2021 đến 2022
rev_2021 = full_sales_df[full_sales_df['Date'].dt.year == 2021]['Revenue'].mean()
rev_2022 = full_sales_df[full_sales_df['Date'].dt.year == 2022]['Revenue'].mean()
yoy_revenue_growth = rev_2022 / rev_2021  # Tăng trưởng ~12%

cogs_2021 = full_sales_df[full_sales_df['Date'].dt.year == 2021]['COGS'].mean()
cogs_2022 = full_sales_df[full_sales_df['Date'].dt.year == 2022]['COGS'].mean()
yoy_cogs_growth = cogs_2022 / cogs_2021

# Áp dụng 50% mức tăng trưởng lịch sử để làm Extrapolation an toàn cho 2023-2024
rev_growth_rate = 1.0 + (yoy_revenue_growth - 1.0) * 0.5 
cogs_growth_rate = 1.0 + (yoy_cogs_growth - 1.0) * 0.5

print(f"Macro Trend Multipliers - Revenue: {rev_growth_rate:.4f}, COGS: {cogs_growth_rate:.4f}")

sales_df = full_sales_df[full_sales_df['Date'] >= '2020-01-01'].reset_index(drop=True)

sub_df = pd.read_csv(SUBMISSION_PATH)
sub_df['Date'] = pd.to_datetime(sub_df['Date'])

# Create a combined timeline
all_dates = pd.concat([
    sales_df[['Date', 'Revenue', 'COGS']], 
    pd.DataFrame({
        'Date': sub_df['Date'], 
        'Revenue': np.nan, 
        'COGS': np.nan
    })
], ignore_index=True)

# --- 3. FEATURE ENGINEERING ---
print("Extracting Pure Seasonality features...")
def create_time_features(df):
    df = df.copy()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Year'] = df['Date'].dt.year
    
    # Flags
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    
    # Double Day Sale
    df['Is_Double_Day'] = (df['Month'] == df['DayOfMonth']).astype(int)
    
    # Fourier terms
    df['sin_365'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['cos_365'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    
    # --- HOLIDAY FEATURES ---
    solar_holidays = [
        (1, 1), (2, 14), (3, 8), (4, 30), (5, 1), 
        (9, 2), (10, 20), (12, 24), (12, 25)
    ]
    df['Is_Solar_Holiday'] = df.apply(lambda row: 1 if (row['Month'], row['DayOfMonth']) in solar_holidays else 0, axis=1)
    
    df['Is_Black_Friday'] = ((df['Month'] == 11) & 
                             (df['DayOfWeek'] == 4) & 
                             (df['DayOfMonth'] >= 22) & 
                             (df['DayOfMonth'] <= 28)).astype(int)
                             
    hung_kings_dates = pd.to_datetime([
        '2020-04-02', '2021-04-21', '2022-04-10', 
        '2023-04-29', '2024-04-18', '2025-04-07'
    ])
    df['Is_Hung_Kings'] = df['Date'].isin(hung_kings_dates).astype(int)
    
    tet_dates = pd.to_datetime([
        '2020-01-25', '2021-02-12', '2022-02-01', 
        '2023-01-22', '2024-02-10', '2025-01-29'
    ])
    
    def get_days_to_tet(current_date):
        future_tets = tet_dates[tet_dates >= current_date]
        if len(future_tets) == 0: 
            return -1
        days_diff = (future_tets[0] - current_date).days
        return days_diff if days_diff <= 30 else -1

    def get_days_after_tet(current_date):
        past_tets = tet_dates[tet_dates <= current_date]
        if len(past_tets) == 0: 
            return -1
        days_diff = (current_date - past_tets[-1]).days
        return days_diff if days_diff <= 15 else -1

    df['Days_To_Tet'] = df['Date'].apply(get_days_to_tet)
    df['Days_After_Tet'] = df['Date'].apply(get_days_after_tet)
    return df

full_df = create_time_features(all_dates)

targets = ['Revenue', 'COGS']

# Lấy dữ liệu lịch sử (2020-2022) để train
train_df = full_df[full_df['Date'] < '2023-01-01'].copy()

# Train/Val Split
val_start = '2022-01-01'
train_data = train_df[train_df['Date'] < val_start]
val_data = train_df[train_df['Date'] >= val_start]

# Select core features
drop_features = ['Date', 'Revenue', 'COGS']
features = [c for c in train_df.columns if c not in drop_features]

# --- 4. MODEL TRAINING WITH HYPERPARAMETER TUNING ---
models = {}
param_grid = {
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4, 5, 6],            
    'subsample': [0.6, 0.7, 0.8, 0.9],      
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9], 
    'min_child_weight': [3, 5, 7, 10],   
    'gamma': [0, 0.1, 1, 3]
}
n_iter_search = 50 
n_splits = 4 

for target in targets:
    print(f"\n{'='*60}")
    print(f"--- XGBoost Seasonality Memorization for {target} ---")
    print(f"{'='*60}")
    
    X_cv_full = train_data[features].reset_index(drop=True)
    y_cv_full = np.log1p(train_data[target]).reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    best_score = float('inf')
    best_params = None
    best_n_estimators = 500
    
    random.seed(42)
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sampled_params = random.sample(param_combinations, min(n_iter_search, len(param_combinations)))
    
    print(f"Testing {len(sampled_params)} hyperparameter combinations...")
    
    for idx, params in enumerate(sampled_params):
        cv_scores = []
        best_iters = []
        
        for train_idx, val_idx in tscv.split(X_cv_full):
            X_cv_train = X_cv_full.iloc[train_idx]
            y_cv_train = y_cv_full.iloc[train_idx]
            X_cv_val = X_cv_full.iloc[val_idx]
            y_cv_val = y_cv_full.iloc[val_idx]
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=1000, 
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            xgb_model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )
            
            pred_log = xgb_model.predict(X_cv_val)
            rmse = np.sqrt(mean_squared_error(y_cv_val, pred_log))
            cv_scores.append(rmse)
            best_iters.append(getattr(xgb_model, 'best_iteration', 1000))
            
        avg_rmse = np.mean(cv_scores)
        avg_iter = int(np.mean(best_iters))
        
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params
            best_n_estimators = avg_iter
            
        if (idx + 1) % 5 == 0 or (idx + 1) == len(sampled_params):
            print(f"  [Iter {idx+1:03d}/{len(sampled_params)}] Best CV RMSE (log): {best_score:.4f} | Optimal Trees: {best_n_estimators}")
            
    print(f"\n[DONE] Best parameters for {target}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # ---------------------------------------------------------
    # VALIDATION EVALUATION (2022)
    # ---------------------------------------------------------
    print("\nEvaluating on 2022 Validation Data...")
    
    val_xgb = xgb.XGBRegressor(
        n_estimators=best_n_estimators,
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    val_xgb.fit(train_data[features], np.log1p(train_data[target]))
    
    val_pred_log = val_xgb.predict(val_data[features])
    preds_exp = np.expm1(val_pred_log)
    y_val_exp = val_data[target].values
    
    mae = mean_absolute_error(y_val_exp, preds_exp)
    rmse = np.sqrt(mean_squared_error(y_val_exp, preds_exp))
    r2 = r2_score(y_val_exp, preds_exp)
    print(f"Validation 2022 - {target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(val_xgb, max_num_features=15, importance_type='weight')
    plt.title(f'Feature Importance ({target})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'feature_importance_{target}.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # HUẤN LUYỆN FINAL MODEL VÀ SEED ENSEMBLING
    # ---------------------------------------------------------
    print(f"Retraining Final Models with Seed Ensembling on ALL 2020-2022 DATA for {target}...")
    
    ensemble_models = []
    seeds = [42, 123, 2026, 777, 999]
    for s in seeds:
        final_xgb = xgb.XGBRegressor(
            n_estimators=best_n_estimators, 
            random_state=s,
            n_jobs=-1,
            **best_params
        )
        final_xgb.fit(train_df[features], np.log1p(train_df[target]))
        ensemble_models.append(final_xgb)
        
    models[target] = ensemble_models

# --- 5. DIRECT FORECASTING & TREND EXTRAPOLATION ---
print("\nStarting DIRECT Forecasting for 2023-2024 with Trend Extrapolation & Seed Ensembling...")

test_df = full_df[full_df['Date'] >= '2023-01-01'].copy()
predictions = {'Date': test_df['Date'].tolist()}

for target in targets:
    # 1. Dự báo Baseline bằng Seed Ensembling
    ensemble_preds_log = np.zeros(len(test_df))
    for model in models[target]:
        ensemble_preds_log += model.predict(test_df[features])
    
    ensemble_preds_log /= len(models[target])
    pred_val = np.expm1(ensemble_preds_log)
    
    # 2. Extrapolate Trend
    growth_rate = rev_growth_rate if target == 'Revenue' else cogs_growth_rate
    
    mask_2023 = test_df['Date'].dt.year == 2023
    mask_2024 = test_df['Date'].dt.year == 2024
    
    # 2023 tăng trưởng 1 bậc
    pred_val[mask_2023] = pred_val[mask_2023] * growth_rate
    # 2024 tăng trưởng 2 bậc
    pred_val[mask_2024] = pred_val[mask_2024] * (growth_rate ** 2)
    
    predictions[target] = pred_val

pred_df = pd.DataFrame(predictions)

# --- 6. SUBMISSION FORMATTING ---
print("\nSaving submission file...")
final_sub = pd.merge(sub_df[['Date']], pred_df, on='Date', how='left')
final_sub.to_csv(OUTPUT_SUB_PATH, index=False)
print(f"Done! Submission saved to {OUTPUT_SUB_PATH}")
print(f"Plots saved to {PLOT_DIR}/ directory.")
