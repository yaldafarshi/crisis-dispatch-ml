"""
Reusable preprocessing pipeline (DataPreprocessingExpert).

Handles missing-value imputation, ordinal/one-hot encoding, and an
80/20 stratified train/test split, with all preprocessing parameters
fit exclusively on the training partition to avoid data leakage.
Also produces quick EDA plots (target distribution / class balance).

Run:
    python preprocessing/preprocessing_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set_theme(style="whitegrid", font_scale=1.1)


class DataPreprocessingExpert:
    def __init__(self, file_path, dataset_type='auto'):
        self.file_path = file_path
        self.df = None
        self.original_shape = None
        self.is_regression = False
        self.target_col = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.dataset_type = dataset_type

        # preprocessing state
        self.numeric_imputers = {}
        self.cat_imputers = {}
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_names_after_onehot = []

        # columns that are already one-hot encoded in the raw data
        self.pre_encoded_cols = [c for c in ['day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                                              'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
                                              'season_1', 'season_2', 'season_3', 'season_4',
                                              'fault_type_1', 'fault_type_2', 'fault_type_3', 'fault_type_4', 'fault_type_5']]

        # ordinal columns
        self.ordinal_cols = ['traffic_level', 'road_quality']

    def load_and_detect_type(self):
        """Load the data and auto-detect the problem type via the target column."""
        print("=" * 60)
        print(f"LOADING: {self.file_path}")
        print("=" * 60)
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File '{self.file_path}' not found!")

            self.df = pd.read_csv(self.file_path)
            self.df.columns = self.df.columns.str.strip()
            self.original_shape = self.df.shape

            potential_targets = ['target_total_time', 'target_team_id']

            found_target = None
            for target in potential_targets:
                matching_cols = [col for col in self.df.columns if col == target]
                if matching_cols:
                    found_target = matching_cols[0]
                    break

            if not found_target:
                print("WARNING: Target column not found by name. Assuming last column is target.")
                found_target = self.df.columns[-1]
                print(f"Assumed Target Column: '{found_target}'")
            else:
                print(f"Found Target Column: '{found_target}'")

            self.is_regression = 'target_total_time' in found_target
            self.target_col = found_target
            print(f">>> Problem Type: {'Regression' if self.is_regression else 'Classification'}")
            print(f">>> Shape: {self.original_shape}")
            print(f"First 5 columns: {list(self.df.columns[:5])}")

            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def identify_columns(self):
        """Identify numeric vs. categorical columns."""
        print("\n" + "=" * 60)
        print("COLUMN IDENTIFICATION")
        print("=" * 60)

        exclude_cols = [
            self.target_col,
            'incident_id',
            'lat_failure',
            'lon_failure'
        ]

        cols = [c for c in self.df.columns if c not in exclude_cols]

        self.numeric_cols = self.df[cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df[cols].select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numeric Features: {len(self.numeric_cols)}")
        print(f"Categorical Features: {len(self.categorical_cols)}")
        return True

    def fit_preprocessing(self, df):
        """Fit on training data only."""
        print("\n[Step 1] Fitting Preprocessing Parameters (on TRAINING Data)...")
        df = df.copy()

        # numeric imputation
        for col in self.numeric_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    self.numeric_imputers[col] = median_val
                    df[col].fillna(median_val, inplace=True)
                else:
                    self.numeric_imputers[col] = df[col].median()

        # categorical imputation
        for col in self.categorical_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    self.cat_imputers[col] = mode_val
                    df[col].fillna(mode_val, inplace=True)
                else:
                    self.cat_imputers[col] = df[col].mode()[0] if not df[col].mode().empty else "Unknown"

        # encoding
        ordinal_cols_in_data = [c for c in self.ordinal_cols if c in df.columns]
        nominal_cols_in_data = [c for c in self.categorical_cols if c not in self.ordinal_cols and c not in self.pre_encoded_cols]

        for col in ordinal_cols_in_data:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.label_encoders[col] = le
            df[col] = le.transform(df[col])

        if nominal_cols_in_data:
            cols_to_onehot = [c for c in nominal_cols_in_data if c in df.columns]
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_data = self.onehot_encoder.fit_transform(df[cols_to_onehot])
            feature_names = self.onehot_encoder.get_feature_names_out(cols_to_onehot)
            self.feature_names_after_onehot = feature_names.tolist()
            df.drop(columns=cols_to_onehot, inplace=True)
            df_new = pd.DataFrame(onehot_data, columns=self.feature_names_after_onehot, index=df.index)
            df = pd.concat([df, df_new], axis=1)
        else:
            self.onehot_encoder = None
            self.feature_names_after_onehot = []

        final_feature_cols = self.numeric_cols.copy()
        final_feature_cols.extend(ordinal_cols_in_data)
        if self.feature_names_after_onehot:
            final_feature_cols.extend(self.feature_names_after_onehot)

        if self.target_col in final_feature_cols:
            final_feature_cols.remove(self.target_col)

        self.final_feature_cols = final_feature_cols

        return df

    def transform_preprocessing(self, df):
        """Transform test/validation data using parameters fit on training data."""
        print("\n[Step 2] Transforming Data (using TRAINING Parameters)...")
        df = df.copy()

        for col in self.numeric_cols:
            if col in df.columns:
                if col in self.numeric_imputers:
                    df[col].fillna(self.numeric_imputers[col], inplace=True)
                else:
                    df[col].fillna(0, inplace=True)

        for col in self.categorical_cols:
            if col in df.columns:
                if col in self.cat_imputers:
                    df[col].fillna(self.cat_imputers[col], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)

        for col in self.ordinal_cols:
            if col in df.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                else:
                    df[col] = 0

        if self.onehot_encoder is not None:
            nominal_cols_in_data = [c for c in self.categorical_cols if c not in self.ordinal_cols and c not in self.pre_encoded_cols]
            cols_to_onehot = [c for c in nominal_cols_in_data if c in df.columns]

            if cols_to_onehot:
                onehot_data = self.onehot_encoder.transform(df[cols_to_onehot])
                df_new = pd.DataFrame(onehot_data, columns=self.feature_names_after_onehot, index=df.index)
                df.drop(columns=cols_to_onehot, inplace=True)
                df = pd.concat([df, df_new], axis=1)

        for col in self.final_feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[self.final_feature_cols]

        return df

    def run_full_pipeline_with_split(self, test_size=0.2, random_state=42):
        """Run the full pipeline: load, split, fit, transform."""
        print("\n" + "=" * 60)
        print(f"STARTING PIPELINE: {self.file_path}")
        print("=" * 60)

        if not self.load_and_detect_type():
            return
        if not self.identify_columns():
            return

        dataset_name = "Reg" if self.is_regression else "Cls"

        # stratify by target for classification problems
        stratify_param = None
        if not self.is_regression and self.target_col in self.df.columns:
            stratify_param = self.df[self.target_col]

        try:
            X_train_df, X_test_df = train_test_split(
                self.df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_param
            )
        except Exception as e:
            print(f"Error in train_test_split: {e}")
            return None, None, None, None

        print(f"Train Shape: {X_train_df.shape}, Test Shape: {X_test_df.shape}")

        print("\n[Step 1] Fitting on TRAINING set...")
        X_train_processed = self.fit_preprocessing(X_train_df)

        print("\n[Step 2] Transforming TEST set...")
        X_test_processed = self.transform_preprocessing(X_test_df)

        if self.target_col in X_train_processed.columns:
            y_train = X_train_processed[self.target_col]
        elif self.target_col in X_train_df.columns:
            y_train = X_train_df[self.target_col]
        else:
            raise ValueError(f"Target column '{self.target_col}' not found in training data after split.")

        if self.target_col in X_test_processed.columns:
            y_test = X_test_processed[self.target_col]
        elif self.target_col in X_test_df.columns:
            y_test = X_test_df[self.target_col]
        else:
            raise ValueError(f"Target column '{self.target_col}' not found in test data after split.")

        features_to_drop = [self.target_col]
        X_train_features = X_train_processed.drop(columns=[c for c in features_to_drop if c in X_train_processed.columns])
        X_test_features = X_test_processed.drop(columns=[c for c in features_to_drop if c in X_test_processed.columns])

        print(f"Final Features Shape (Train): {X_train_features.shape}")
        print(f"Final Features Shape (Test): {X_test_features.shape}")

        self.X_train = X_train_features.values
        self.y_train = y_train.values
        self.X_test = X_test_features.values
        self.y_test = y_test.values

        self.run_eda_on_train(X_train_processed, dataset_name)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def run_eda_on_train(self, df, dataset_name):
        print("\n[EDA] Running EDA on Training Data...")
        if self.is_regression:
            self.plot_target_dist(df, dataset_name)
        else:
            self.plot_class_balance(df, dataset_name)

    def plot_target_dist(self, df, name):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[self.target_col].dropna(), kde=True, color='purple', bins=30)
        plt.title(f'Target Distribution ({name})')
        plt.tight_layout()
        plt.savefig(f'target_{name}.png')
        plt.show()

    def plot_class_balance(self, df, name):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[self.target_col], palette='viridis')
        plt.title(f'Class Balance ({name})')
        plt.tight_layout()
        plt.savefig(f'class_{name}.png')
        plt.show()


if __name__ == "__main__":
    print("\n>>> RUNNING FOR REGRESSION DATASET <<<\n")
    try:
        proc_reg = DataPreprocessingExpert('crisis_dataset_regression_v52.csv', 'regression')
        X_train_reg, y_train_reg, X_test_reg, y_test_reg = proc_reg.run_full_pipeline_with_split()
        if X_train_reg is not None:
            print("Regression Pipeline Completed Successfully.")
        else:
            print("Regression Pipeline Failed.")
    except Exception as e:
        print(f"Error in Regression Pipeline: {e}")

    print("\n\n>>> RUNNING FOR CLASSIFICATION DATASET <<<\n")
    try:
        proc_cls = DataPreprocessingExpert('crisis_dataset_classifier_v52.csv', 'classification')
        X_train_cls, y_train_cls, X_test_cls, y_test_cls = proc_cls.run_full_pipeline_with_split()
        if X_train_cls is not None:
            print("Classification Pipeline Completed Successfully.")
        else:
            print("Classification Pipeline Failed.")
    except Exception as e:
        print(f"Error in Classification Pipeline: {e}")
