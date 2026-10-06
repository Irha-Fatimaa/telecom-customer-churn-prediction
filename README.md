# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. The repository covers data generation, cleaning, feature engineering, model selection, evaluation, threshold optimization, model persistence, command-line inference, and Streamlit deployment.

## Project Structure

```text
telecom-customer-churn/
├── app.py                         # Streamlit deployment app
├── predict.py                     # Command-line single-customer prediction
├── generate_dataset.py            # Reproducible telecom dataset generator
├── requirements.txt
├── Dockerfile
├── Procfile
├── LICENSE
├── SUBMISSION_CHECKLIST.md
├── data/
│   └── telecom_customer_churn.csv
├── src/
│   ├── features.py                # Data cleaning + feature engineering
│   └── train.py                   # Model training, tuning and evaluation
├── models/
│   ├── churn_model.joblib         # Trained sklearn pipeline
│   └── metadata.json              # Threshold, model name and columns
├── reports/
│   ├── Churn_Analysis_Report.pdf
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   ├── feature_importance.csv
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       ├── threshold_optimization.png
│       ├── feature_importance.png
│       └── model_comparison.png
├── screenshots/                   # Submission-ready evidence images
└── Customer_Churn_Analysis.ipynb  # Notebook walkthrough
```

## Dataset

The included dataset contains **6,000 synthetic but realistic telecom customer records**. It is generated with a fixed random seed (`42`) so the project is fully reproducible and does not depend on a Kaggle login or external download.

It follows the schema commonly used for telecom churn analysis: demographics, tenure, phone/internet services, support services, contract type, billing/payment method, monthly charges, total charges, and a binary `Churn` target.

Why synthetic? It makes the submission self-contained and reproducible for grading. The code architecture also works with a real telco churn CSV if the same fields are provided.

## Feature Engineering

The project adds business-relevant features before modeling:

1. `NumServices` - number of subscribed telecom services.
2. `HasInternet` - whether an internet service is active.
3. `HasSupportBundle` - whether both Online Security and Tech Support are active.
4. `AutoPay` - whether the customer uses automatic bank/card payment.
5. `FamilyAccount` - whether partner or dependents are present.
6. `AvgMonthlySpend` - total charges divided by tenure.
7. `ChargeIncreaseRatio` - current monthly charge relative to historical average spend.
8. `TenureBand` - customer lifecycle buckets from new to long-term.
9. `HighMonthlyCharge` - indicator for monthly charges >= 90.
10. `MonthToMonth` - direct churn-risk indicator for short contracts.
11. `SupportGap` - internet customer without adequate support/security coverage.

All preprocessing is inside a scikit-learn `Pipeline` / `ColumnTransformer`, reducing leakage risk and ensuring deployment uses exactly the same transformations as training.

## Model Selection

Three baseline models are compared:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

A compact manual tuning step evaluates multiple Random Forest configurations on a validation split. The final selected model is chosen by held-out ROC-AUC.

### Actual Results From This Package

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| Test ROC-AUC | 0.7864 |
| Test PR-AUC | 0.7290 |
| Tuned threshold | 0.33 |
| Accuracy at tuned threshold | 0.7017 |
| Precision at tuned threshold | 0.6263 |
| Recall at tuned threshold | 0.8889 |
| F1 at tuned threshold | 0.7348 |
| Accuracy at default 0.50 threshold | 0.7358 |
| F1 at default 0.50 threshold | 0.7246 |

The deployment threshold is intentionally tuned toward **high recall**. In a churn-retention problem, failing to identify an actual churner can be more expensive than contacting an additional lower-risk customer. The selected threshold raises churn recall to ~88.9% while slightly reducing overall accuracy.

## How to Run Locally (Windows)

### 1. Extract the ZIP
Open PowerShell or Command Prompt inside the project folder.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Recreate the dataset (optional)
The CSV is already included, but you can regenerate it exactly:

```bash
python generate_dataset.py
```

### 5. Retrain and evaluate

```bash
set PYTHONPATH=src
python src\train.py
```

PowerShell alternative:

```powershell
$env:PYTHONPATH="src"
python src/train.py
```

This regenerates the model, metrics, CSV reports, and figures.

### 6. Test a prediction

```bash
python predict.py
```

Expected behavior: it prints a churn/stay prediction, churn probability, and decision threshold.

### 7. Launch the deployment app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## Deployment Options

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

### Docker

```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Then open `http://localhost:8501`.

## Evaluation Artifacts

The `reports/figures/` folder contains:

- **Confusion matrix** - summarizes correct and incorrect churn decisions.
- **ROC curve** - shows discrimination over all thresholds.
- **Precision-Recall curve** - especially useful for churn-class performance.
- **Threshold optimization** - shows precision, recall and F1 across cutoffs.
- **Feature importance** - permutation-based business driver ranking.
- **Model comparison** - cross-validated ROC-AUC across candidate algorithms.

## Business Interpretation

The engineered and raw features allow the model to prioritize actionable risk patterns. Important signals in this run include month-to-month contracts, tenure, internet service, contract type, total charges, and support-related services. A telecom retention team could use the churn probability to segment customers into outreach priorities rather than relying on a single yes/no rule.

Suggested operational use:

- High risk: immediate retention campaign or personalized offer.
- Medium risk: targeted engagement and service-quality follow-up.
- Low risk: normal customer relationship management.

## Reproducibility

- Random seed: `42`
- Included dataset: yes
- Included trained model: yes
- Included requirements: yes
- Included saved metrics and figures: yes
- Training pipeline avoids fitting preprocessing on the full dataset before splitting.

## Files to Submit

For the BeeNeural task submission, use:

1. GitHub repository link containing the complete project.
2. `reports/Churn_Analysis_Report.pdf`.
3. Screenshots from `screenshots/` plus a screenshot of the running Streamlit page after you launch it locally.
4. The complete ZIP as an uploaded supporting file if the platform allows it.

## Notes

This project is intended as an academic machine-learning submission and demonstration. Model performance on a synthetic benchmark does not guarantee the same performance on live telecom customers; a production system would require real labeled data, monitoring, drift detection, fairness checks, privacy review, and periodic retraining.
