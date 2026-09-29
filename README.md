# Intelligent Crisis-Dispatch System for Power Distribution Networks

A dual machine-learning framework that replaces manual repair-crew dispatch
in electrical power distribution networks with two supervised models:

1. **Classifier** — recommends the optimal repair team (of 5 candidates) for
   an incoming fault.
2. **Regressor** — estimates the resulting total repair time (travel + repair,
   in minutes).

Course project for *Machine Learning*.

The full write-up, with methodology, equations, and figures, is in
[`docs/Intelligent_Crisis_Dispatch_System_Essay.docx`](docs/Intelligent_Crisis_Dispatch_System_Essay.docx).

## Summary of results

| Task                        | Model         | Train R² / Acc | Test R² / Acc | Notes                          |
|------------------------------|---------------|-----------------|-----------------|---------------------------------|
| Repair-time regression        | Random Forest | 0.865           | 0.865           | Stable — **selected**           |
| Repair-time regression        | XGBoost       | 0.909           | 0.534           | Overfits — kept for comparison  |
| Optimal-team classification   | Random Forest | —               | 83%             |                                  |
| Optimal-team classification   | XGBoost       | —               | **93%**         | **Selected**                    |

No real utility data was available, so a physics- and rule-based simulator
generates 1,997 labeled incidents (52 engineered features) around five team
depots in Tehran, using the Haversine distance, seasonal/hourly weather,
per-team skill matrices, and workload-based availability.

## Project structure

```
crisis-dispatch-ml/
├── data_generation/
│   └── generate_dataset.py            # simulates the 1,997-incident dataset
├── preprocessing/
│   └── preprocessing_pipeline.py      # DataPreprocessingExpert: impute/encode/split, leakage-free
├── src/
│   ├── classification/
│   │   ├── train_random_forest.py     # team-selection model, 83% accuracy
│   │   └── train_xgboost.py           # team-selection model, 93% accuracy (selected)
│   ├── regression/
│   │   ├── train_random_forest.py     # repair-time model, R² 0.865 (selected)
│   │   ├── train_xgboost.py           # repair-time model, R² 0.909 train / 0.534 test (overfits)
│   │   └── evaluate_regressors.py     # head-to-head RF vs XGBoost comparison on a held-out set
│   └── utils/
│       └── make_test_split.py         # re-simulates an unseen held-out set for evaluate_regressors.py
├── docs/
│   └── Intelligent_Crisis_Dispatch_System_Essay.docx
├── outputs/                           # saved plots land here when you run the scripts
├── models/                            # saved .pkl models land here when you run the scripts
└── requirements.txt
```

> **Note:** `evaluate_regressors.py` expects a `test_reg_data.csv` held-out
> file and reads `scaler.pkl` / `feature_cols.pkl` saved by
> `src/regression/train_random_forest.py`. Run `src/utils/make_test_split.py`
> first — it re-simulates a fresh batch of incidents with a different random
> seed, so the held-out set is genuinely unseen data, not a re-split of the
> same 2,000 incidents the models trained on.

## Getting started

```bash
git clone <your-repo-url>
cd crisis-dispatch-ml
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python data_generation/generate_dataset.py

# 2. (optional) Explore the preprocessing pipeline / EDA plots
python preprocessing/preprocessing_pipeline.py

# 3. Train the classifiers
python src/classification/train_random_forest.py
python src/classification/train_xgboost.py

# 4. Train the regressors
python src/regression/train_random_forest.py
python src/regression/train_xgboost.py

# 5. Generate an independent held-out test set (unseen during training)
python src/utils/make_test_split.py

# 6. Compare the two regressors on that held-out set
python src/regression/evaluate_regressors.py
```

> **Note:** the scripts currently save models/plots to the working directory
> they're run from (`*.pkl`, `*.png`). Feel free to point `joblib.dump(...)`
> and `plt.savefig(...)` calls at `models/` and `outputs/` respectively if
> you want everything organized on disk automatically.

## Method highlights

- **Data**: 2,000 simulated incidents (1,997 after removing unavailable-team
  cases), 52 features across spatial/temporal, environmental, fault/equipment,
  and per-team families.
- **Labels**: an explicit cost formula (distance × 2 + skill-inefficiency +
  weather penalty + workload × 5) defines the "optimal" team for 90% of
  incidents; the other 10% simulate an inconsistent human dispatcher. The
  classification scripts re-derive this cost formula directly and correct
  any incident where the dispatched team wasn't actually the optimal one.
- **Preprocessing**: median/mode imputation, one-hot encoding for nominal
  variables, ordinal encoding for `traffic_level` / `road_quality`,
  `StandardScaler` for numeric features, an 80/20 stratified split with all
  parameters fit only on the training partition. The repair-time target is
  log-transformed (`log1p`) since it's right-skewed.
- **Models**: Random Forest (Breiman, 2001) vs. XGBoost (Chen & Guestrin,
  2016) for both tasks, to contrast bagging vs. gradient boosting.
- **Key finding**: XGBoost is the stronger classifier (93% vs. 83%) but the
  weaker regressor once train/test performance are compared (0.909 train R²
  vs. 0.534 test R²) — a concrete demonstration of why generalization, not
  training-set performance, should drive model selection.

## Requirements

See [`requirements.txt`](requirements.txt). Python 3.9+ recommended.

## License

Add a license of your choice (e.g. MIT) before publishing, if you want
others to be able to reuse the code.
