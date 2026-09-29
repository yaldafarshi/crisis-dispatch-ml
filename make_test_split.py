"""
Builds an independent held-out test set for evaluate_regressors.py.

The training scripts (src/regression/train_random_forest.py and
train_xgboost.py) already carve an 80/20 split out of
crisis_dataset_regression_v52.csv, so re-using that same 20% here
would not really test generalization -- the models have already been
scored on it once.

Instead, this script re-runs the same simulator with a different
random seed, producing a fresh batch of incidents the models have
never seen at all, and saves it as test_reg_data.csv for
evaluate_regressors.py to score.

Run:
    python src/utils/make_test_split.py
"""

import sys
import os

# allow importing the sibling data_generation package
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from data_generation.generate_dataset import generate_crisis_dataset

# a different seed from the one used for the training data (42),
# so this is genuinely unseen data, not a re-split of the same set
HOLDOUT_SEED = 99
N_HOLDOUT_SAMPLES = 400

if __name__ == "__main__":
    print(f"Generating an independent held-out set (seed={HOLDOUT_SEED})...")

    np.random.seed(HOLDOUT_SEED)
    df_holdout = generate_crisis_dataset(
        n_samples=N_HOLDOUT_SAMPLES,
        n_teams=5,
        n_fault_types=5
    )

    # evaluate_regressors.py only needs the regression columns
    # (team_id is only relevant for the classification task)
    df_holdout_reg = df_holdout.drop(columns=['target_team_id'])

    out_path = 'test_reg_data.csv'
    df_holdout_reg.to_csv(out_path, index=False)

    print(f"Saved {len(df_holdout_reg)} held-out incidents to '{out_path}'.")
    print("You can now run: python src/regression/evaluate_regressors.py")
