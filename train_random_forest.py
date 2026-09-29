"""
Repair-time regressor (Model 1a): Random Forest.

Trains a RandomForestRegressor on crisis_dataset_regression_v52.csv to
predict target_total_time (log1p-transformed). This is the final model
selected in the write-up, since its R^2 (~0.865) is stable between
train and test, unlike the XGBoost regressor which overfits (see
train_xgboost.py and evaluate_regressors.py).

Saves: best_rf_regressor.pkl, scaler.pkl, feature_cols.pkl

Run:
    python src/regression/train_random_forest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import os

warnings.filterwarnings('ignore')
pd.set_option('display.precision', 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set(style="whitegrid")


# ==============================================================================
# 1. Helper functions (data preprocessing)
# ==============================================================================
def clean_data(filepath):
    """Load, clean, and lightly prepare the raw CSV."""
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # 1. clean infinities / invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 2. convert categorical variables to numeric
    if 'weather_condition' in df.columns:
        if df['weather_condition'].dtype == 'object':
            weather_map = {'good': 0, 'rain': 1, 'fog': 2, 'storm': 3}
            df['weather_condition'] = df['weather_condition'].map(weather_map)

    for col in df.select_dtypes(include=['object']).columns:
        # convert any remaining text columns to numeric codes, except the targets
        if col not in ['target_team_id', 'target_total_time']:
            df[col] = df[col].astype('category').cat.codes

    return df


def prepare_features_and_target(df):
    """
    Final feature (X) / target (y) preparation for the v52 dataset.
    Target: predict the total repair time of the dispatched team
    (target_total_time).
    """
    target_col = 'target_total_time'

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                          f"Available columns: {list(df.columns)}")

    # columns that should never be used as model input
    exclude_cols = [
        'incident_id',
        'target_team_id',
        'lat_failure',
        'lon_failure',
        target_col
    ]

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude_cols]

    # log-transform the target (it's right-skewed) to help the model fit
    y_original = df[target_col]
    y_log = np.log1p(y_original)  # log(1 + x), avoids log(0)

    X = df[feature_cols]

    print(f"Total features (X): {len(feature_cols)}")
    print(f"Target column: {target_col}")
    print(f"Sample features: {feature_cols[:5]}...")

    return X, y_log, feature_cols


# ==============================================================================
# 2. Training class
# ==============================================================================
class TimeRepairModelTrainer:
    def __init__(self, filepath='crisis_dataset_regression_v52.csv'):
        self.filepath = filepath
        self.X = None
        self.y_log = None
        self.y_original_test = None
        self.y_pred_original_test = None  # stored for plotting
        self.feature_cols = None
        self.model = None
        self.scaler = StandardScaler()

    def run_pipeline(self):
        print("=" * 60)
        print("STEP 1: DATA LOADING & PREPARATION (v52)")
        print("=" * 60)

        df = clean_data(self.filepath)
        self.X, self.y_log, self.feature_cols = prepare_features_and_target(df)

        print("\n" + "=" * 60)
        print("STEP 2: SPLITTING & SCALING")
        print("=" * 60)

        X_train, X_test, y_train_log, y_test_log = train_test_split(
            self.X, self.y_log, test_size=0.2, random_state=42
        )

        # keep the un-logged values for final evaluation
        self.y_original_test = np.expm1(y_test_log.values)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")

        print("\n" + "=" * 60)
        print("STEP 3: MODEL TRAINING (Random Forest Regressor)")
        print("=" * 60)

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train_log)

        print("\n" + "=" * 60)
        print("STEP 4: EVALUATION")
        print("=" * 60)

        y_pred_log = self.model.predict(X_test_scaled)
        self.y_pred_original_test = np.expm1(y_pred_log)

        rmse = np.sqrt(mean_squared_error(self.y_original_test, self.y_pred_original_test))
        mae = mean_absolute_error(self.y_original_test, self.y_pred_original_test)
        r2 = r2_score(self.y_original_test, self.y_pred_original_test)

        # MAPE, guarding against values near zero
        mask = self.y_original_test > 0.1
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(
                self.y_original_test[mask],
                self.y_pred_original_test[mask]
            ) * 100
        else:
            mape = 0

        print(f"   RMSE: {rmse:.2f} minutes")
        print(f"   MAE:  {mae:.2f} minutes")
        print(f"   R^2:  {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")

        print("\n" + "-" * 60)
        print("INTERPRETATION:")
        print("-" * 60)
        print(f"   1. R^2 of {r2:.2f} means the model explains {r2 * 100:.1f}% of the variance.")
        print(f"   2. RMSE of {rmse:.2f} means the average prediction error is ~{rmse:.1f} minutes.")
        print("   3. The model learns from physical features (distance, skill, weather) with no data leakage.")

        self.analyze_features()
        self.plot_results()

        print("\n" + "=" * 60)
        print("SAVING MODEL & SCALER")
        print("=" * 60)

        joblib.dump(self.model, 'best_rf_regressor.pkl')
        print("Model saved as 'best_rf_regressor.pkl'")

        joblib.dump(self.scaler, 'scaler.pkl')
        print("Scaler saved as 'scaler.pkl'")

        joblib.dump(self.feature_cols, 'feature_cols.pkl')
        print("Feature columns saved as 'feature_cols.pkl'")

        return rmse, mae, r2, mape

    def plot_results(self):
        """Error-analysis plots (actual vs. predicted, residual distribution)."""
        print("\n" + "=" * 60)
        print("STEP 5: VISUALIZATION (Actual vs Predicted & Residuals)")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(self.y_original_test, self.y_pred_original_test, alpha=0.5, color='steelblue', edgecolors='k')
        min_val = min(self.y_original_test.min(), self.y_pred_original_test.min())
        max_val = max(self.y_original_test.max(), self.y_pred_original_test.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Total Time (min)')
        axes[0].set_ylabel('Predicted Total Time (min)')
        axes[0].set_title('Actual vs Predicted Repair Time')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        residuals = self.y_original_test - self.y_pred_original_test
        axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[1].set_xlabel('Residual (Error)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Prediction Errors')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('regression_analysis_v52.png', dpi=300)
        print("Plot saved as 'regression_analysis_v52.png'")
        plt.show()

    def analyze_features(self):
        print("\n" + "=" * 60)
        print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        top_10 = importance_df.head(10)

        plt.figure(figsize=(12, 6))
        plt.title('Top 10 Features (v52, no leakage)')
        plt.bar(range(10), top_10['Importance'].values, align='center', color='skyblue')
        plt.xticks(range(10), top_10['Feature'].values, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig('feature_importance_v52.png', dpi=300)
        print("Plot saved as 'feature_importance_v52.png'")
        plt.show()

        print("\nTop 10 features:")
        for idx, row in top_10.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")

        key_features = [
            'team_1_skill', 'team_1_dist', 'weather_condition',
            'crisis_score', 'equipment_age'
        ]

        print("\nKey feature performance:")
        for kf in key_features:
            if kf in importance_df['Feature'].values:
                imp = importance_df[importance_df['Feature'] == kf]['Importance'].values[0]
                rank = importance_df[importance_df['Feature'] == kf].index[0] + 1
                print(f"   - {kf}: rank #{rank}, importance {imp:.4f}")
            else:
                print(f"   - {kf}: not in top features.")


# ==============================================================================
# 3. Entry point
# ==============================================================================
if __name__ == "__main__":
    filepath = 'crisis_dataset_regression_v52.csv'

    if not os.path.exists(filepath):
        print(f"ERROR: File '{filepath}' not found.")
        print("Please run the data generation script first.")
    else:
        trainer = TimeRepairModelTrainer(filepath=filepath)
        trainer.run_pipeline()
