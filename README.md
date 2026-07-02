# Credit Card Fraud Detection — Where Accuracy Is a Lie

**284,807 transactions. 492 frauds. A model that predicts "never fraud" scores 99.83% accuracy — so this project is about the metrics that can't be gamed.**

A Random Forest classifier on the Kaggle Credit Card Fraud dataset (PCA-anonymized features V1–V28 plus Time/Amount, fraud rate 0.17%), evaluated the way an extremely imbalanced problem demands.

## Results (20% held-out test set, from actual notebook output)

| Metric | Value | What it means here |
|---|---|---|
| Accuracy | 0.9996 | Nearly meaningless at 0.17% fraud — reported only to make the point |
| **Precision** | **0.974** | When the model flags fraud, it's right 97% of the time — few false alarms |
| **Recall** | **0.765** | It catches 3 of every 4 frauds — **and misses 1 in 4** |
| F1 | 0.857 | Balance of the two |
| **MCC** | **0.863** | Matthews correlation — the single most honest summary on imbalanced data |

The trade-off stated plainly: this model is conservative. High precision means investigators aren't drowning in false positives, but a quarter of fraud slips through. In production you'd tune the decision threshold along the precision/recall curve based on what a missed fraud costs versus what an investigation costs — that costing decision is a business call, not a modeling one.

## What the notebook does

1. Loads and profiles the data; quantifies the imbalance (492 fraud vs. 284,315 valid) and compares amount distributions between classes.
2. Correlation heatmap across features.
3. Trains a Random Forest on an 80/20 split — deliberately baseline: no SMOTE, no threshold tuning, no hyperparameter search.
4. Evaluates with precision/recall/F1/MCC and a confusion matrix.

Honest scope note: this is a baseline learning project. The obvious next steps — class-weighting or resampling, threshold tuning against a cost matrix, and precision-recall-curve analysis — are named here rather than claimed.

## Run

```
pip install -r requirements.txt
jupyter notebook main.ipynb
```

Requires `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) in the same directory (not committed — size/licensing).

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
