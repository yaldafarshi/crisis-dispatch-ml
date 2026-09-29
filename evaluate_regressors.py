"""
Head-to-head evaluation of the saved Random Forest and XGBoost
regressors on a held-out test CSV (see src/utils/make_test_split.py --
not included in this repo yet).

Loads best_rf_regressor.pkl / best_xgb_regressor.pkl + scaler.pkl +
feature_cols.pkl, scores both models (RMSE, MAE, R^2, MAPE), and
reports the winner by RMSE.

Run:
    python src/regression/evaluate_regressors.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.precision', 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set(style="whitegrid")


# ==============================================================================
# 1. Helper functions (mirrors the training scripts, for consistency)
# ==============================================================================
def clean_data(filepath):
    """Load, clean, and lightly prepare the raw CSV."""
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if 'weather_condition' in df.columns:
        if df['weather_condition'].dtype == 'object':
            weather_map = {'good': 0, 'rain': 1, 'fog': 2, 'storm': 3}
            df['weather_condition'] = df['weather_condition'].map(weather_map)

    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['target_team_id', 'target_total_time']:
            df[col] = df[col].astype('category').cat.codes

    return df


def prepare_features_and_target_for_eval(df, feature_cols):
    """
    Prepare features and target for evaluation. Uses the saved
    feature_cols list so column selection and order exactly match
    what the model was trained on.
    """
    target_col = 'target_total_time'

    X = df[feature_cols]

    y_original = df[target_col]
    y_log = np.log1p(y_original)

    return X, y_log, y_original


# ==============================================================================
# 2. Evaluation class
# ==============================================================================
class RegressorEvaluator:
    def __init__(self, test_filepath, model_paths):
        """
        model_paths: dict mapping a model name to its saved path,
        e.g. {'rf': 'best_rf_regressor.pkl', 'xgb': 'best_xgb_regressor.pkl'}
        """
        self.test_filepath = test_filepath
        self.model_paths = model_paths
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        self.X = None
        self.X_scaled = None
        self.y_log = None
        self.y_original = None

    def load_models_and_scaler(self):
        print("=" * 60)
        print("LOADING MODELS & SCALER")
        print("=" * 60)

        try:
            self.scaler = joblib.load('scaler.pkl')
            self.feature_cols = joblib.load('feature_cols.pkl')
            print("Scaler and feature columns loaded.")
        except FileNotFoundError:
            print("WARNING: 'scaler.pkl' or 'feature_cols.pkl' not found.")
            print("Copy these files over from the training step before running this script.")
            raise FileNotFoundError("Missing scaler.pkl / feature_cols.pkl -- run a training script first.")

        for name, path in self.model_paths.items():
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded model '{name}': {path}")
            else:
                print(f"Model '{name}' not found at: {path}")

    def prepare_test_data(self):
        print("\n" + "=" * 60)
        print("PREPARING TEST DATA")
        print("=" * 60)

        df = clean_data(self.test_filepath)
        self.X, self.y_log, self.y_original = prepare_features_and_target_for_eval(df, self.feature_cols)

        # scale with the SAME scaler fit during training (no re-fitting here)
        self.X_scaled = self.scaler.transform(self.X)

        print(f"Test samples: {self.X_scaled.shape[0]}")

    def evaluate_all(self):
        print("\n" + "=" * 60)
        print("EVALUATING MODELS")
        print("=" * 60)

        results = {}

        for name, model in self.models.items():
            print(f"\n--- Evaluating {name} ---")

            y_pred_log = model.predict(self.X_scaled)
            y_pred_original = np.expm1(y_pred_log)

            rmse = np.sqrt(mean_squared_error(self.y_original, y_pred_original))
            mae = mean_absolute_error(self.y_original, y_pred_original)
            r2 = r2_score(self.y_original, y_pred_original)

            mask = self.y_original > 0.1
            if mask.sum() > 0:
                mape = mean_absolute_percentage_error(
                    self.y_original[mask],
                    y_pred_original[mask]
                ) * 100
            else:
                mape = 0

            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }

            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAE:  {mae:.2f}")
            print(f"   R^2:  {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")

        return results

    def compare_results(self, results):
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<15} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'MAPE':<10}")
        print("-" * 60)

        best_model = None
        best_rmse = float('inf')

        for name, metrics in results.items():
            print(f"{name:<15} | {metrics['RMSE']:<10.2f} | {metrics['MAE']:<10.2f} | "
                  f"{metrics['R2']:<10.4f} | {metrics['MAPE']:<10.2f}%")
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_model = name

        print("-" * 60)
        print(f"Best model (lowest RMSE): {best_model}")
        return best_model


# ==============================================================================
# 3. Entry point
# ==============================================================================
if __name__ == "__main__":
    model_paths = {
        'rf': 'best_rf_regressor.pkl',
        'xgb': 'best_xgb_regressor.pkl'
    }

    # make sure this file has the same feature columns + target_total_time
    test_file = 'test_reg_data.csv'

    evaluator = RegressorEvaluator(test_file, model_paths)
    evaluator.load_models_and_scaler()
    evaluator.prepare_test_data()
    results = evaluator.evaluate_all()
    best_model_name = evaluator.compare_results(results)
