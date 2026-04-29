# ML Competition: Christmas Tree Survival Prediction

This repository contains a machine learning solution for a binary classification competition. The task is to predict whether a Christmas tree will survive until January 18 based on tree characteristics, apartment conditions and care-related factors.

The project is intentionally compact: one training script, competition data files and a reproducible pipeline for generating a submission file.

## Task

The target variable is:

```text
survived_to_18jan
```

The model predicts the probability of the positive class for each object from `test.csv`. The final predictions are saved in the same format as `sample_submission.csv`.

The main validation metric used in the solution is **ROC-AUC**.

## Repository structure

```text
ML-competition/
├── README.md
├── requirements.txt
├── .gitignore
├── train.csv
├── test.csv
├── sample_submission.csv
└── src/
    └── train_catboost_ensemble.py
```

## Approach

The solution is based on CatBoost because the dataset contains both numerical and categorical features. CatBoost can work with categorical columns directly and usually performs well on tabular competition tasks without heavy preprocessing.

The pipeline includes:

- feature generation from the original columns;
- missing-value indicators for important numerical fields;
- domain-inspired features related to heat, humidity, watering and tree age;
- several CatBoost models with different parameters;
- 5-fold stratified cross-validation;
- model blending using weighted, rank-based and logit-based averaging;
- automatic selection of the best blend by out-of-fold ROC-AUC.

## Feature engineering

Several groups of features are created in the script:

- **Tree age and size**: effective tree age by January 18, height in meters, height-based ratios.
- **Heat and dryness**: room temperature, humidity deficit, radiator distance, garland usage and heating type interactions.
- **Care intensity**: watering frequency, mist spraying and watering normalized by tree age.
- **Apartment context**: area, ceiling height, area per person and heat load per area.
- **Interaction features**: combinations of temperature, humidity, garlands, tree form, pets and children.
- **Missing-value flags**: explicit binary indicators showing where the original data had missing values.

These features are not meant to overcomplicate the solution. They describe the same simple idea from different angles: a tree is more likely to dry out if it is old, exposed to heat, poorly watered or placed in an unfavorable environment.

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training script from the repository root:

```bash
python src/train_catboost_ensemble.py --data-dir . --output submission.csv
```

The script expects the following files inside `--data-dir`:

```text
train.csv
test.csv
sample_submission.csv
```

After training, the script creates:

```text
submission.csv
```

## Running in Google Colab

Upload the repository files or clone the repository in Colab, install dependencies and run:

```bash
!pip -q install -r requirements.txt
!python src/train_catboost_ensemble.py --data-dir . --output submission.csv
```

If the CSV files are stored in another folder, pass that folder through `--data-dir`.

## Output

During training, the script prints:

- ROC-AUC for each fold;
- out-of-fold ROC-AUC for each CatBoost model;
- ROC-AUC for different blending strategies;
- the final blending strategy selected for `submission.csv`.

## Main technologies

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost

## Notes

The repository focuses on a clear and reproducible competition solution rather than on a large project architecture. The main goal is to keep the pipeline easy to read, run and modify.