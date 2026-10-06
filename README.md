# Telecom Customer Churn Prediction

A machine learning project for predicting customer churn in a telecommunications setting. The project covers the full workflow from data preparation and feature engineering to model evaluation and deployment through a Streamlit web application.

## Overview

Customer churn is an important business problem for subscription-based companies because retaining an existing customer is often more cost-effective than acquiring a new one. This project builds a binary classification model that estimates the probability that a customer is likely to leave the service.

The final model is a Logistic Regression pipeline trained on customer demographic, account, billing, and service-related information. The prediction threshold was adjusted to improve recall for churners, which is useful in a retention scenario where missing an at-risk customer can be costly.

## Project Workflow

The project follows these main steps:

1. Data generation and loading
2. Data cleaning and preprocessing
3. Feature engineering
4. Train/test splitting
5. Model comparison
6. Model evaluation
7. Decision-threshold analysis
8. Model persistence
9. Streamlit deployment

## Dataset

The repository includes a reproducible telecom customer dataset containing **6,000 records**. The dataset is generated with a fixed random seed so that the same data can be recreated when needed.

The available variables include:

- Customer demographics
- Tenure
- Internet and phone services
- Security and technical support services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

The target variable is `Churn`, where the model predicts whether a customer is likely to leave the service.

## Feature Engineering

In addition to the original variables, several features were created to capture useful customer behavior and account characteristics:

- `NumServices` — total number of subscribed services
- `HasInternet` — whether the customer has an active internet service
- `HasSupportBundle` — whether both online security and technical support are active
- `AutoPay` — whether an automatic payment method is used
- `FamilyAccount` — whether the customer has a partner or dependents
- `AvgMonthlySpend` — historical average monthly spend
- `ChargeIncreaseRatio` — current monthly charge relative to historical average spend
- `TenureBand` — customer tenure grouped into lifecycle ranges
- `HighMonthlyCharge` — indicator for relatively high monthly charges
- `MonthToMonth` — indicator for month-to-month contracts
- `SupportGap` — internet customers without adequate support or security services

Preprocessing is handled inside a scikit-learn pipeline so the same transformations are applied during both training and prediction.

## Models Evaluated

The following classification algorithms were compared:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

The final model was selected using validation and test performance, with ROC-AUC used as one of the main comparison metrics.

## Final Model Performance

The selected model is **Logistic Regression**.

| Metric | Score |
|---|---:|
| ROC-AUC | 0.7864 |
| PR-AUC | 0.7290 |
| Accuracy | 0.7017 |
| Precision | 0.6263 |
| Recall | 0.8889 |
| F1 Score | 0.7348 |
| Decision Threshold | 0.33 |

The threshold of `0.33` was chosen to increase churn recall. At this threshold, the model identifies approximately **88.9% of actual churners** in the test set.

For comparison, the default threshold of `0.50` produced an accuracy of `0.7358` and an F1 score of `0.7246`.

## Evaluation

The repository includes the following evaluation outputs:

- Confusion matrix
- ROC curve
- Precision-Recall curve
- Model comparison chart
- Threshold analysis
- Feature importance analysis
- Classification report

Generated plots and metric files are available in the `reports/` directory.

## Project Structure

```text
telecom-customer-churn-prediction/
├── app.py
├── predict.py
├── generate_dataset.py
├── Customer_Churn_Analysis.ipynb
├── requirements.txt
├── Dockerfile
├── Procfile
├── data/
│   └── telecom_customer_churn.csv
├── src/
│   ├── features.py
│   └── train.py
├── models/
│   ├── churn_model.joblib
│   └── metadata.json
├── reports/
│   ├── Churn_Analysis_Report.pdf
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   ├── feature_importance.csv
│   └── figures/
└── screenshots/
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Irha-Fatimaa/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Test the saved model

```bash
python predict.py
```

Example output:

```text
Prediction: Churn
Churn probability: 93.138%
Decision threshold: 0.33
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Retraining the Model

The included dataset can be regenerated using:

```bash
python generate_dataset.py
```

To retrain the model in PowerShell:

```powershell
$env:PYTHONPATH="src"
python src/train.py
```

This recreates the trained model, evaluation metrics, and plots.

## Streamlit Application

The Streamlit interface allows a user to enter customer information and receive:

- Predicted churn class
- Churn probability
- Risk interpretation
- Decision threshold used by the model

The application uses the same saved preprocessing and prediction pipeline used during model development.

## Docker

The project can also be run in a Docker container:

```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Then open:

```text
http://localhost:8501
```

## Repository Contents

- `Customer_Churn_Analysis.ipynb` — notebook containing the analysis workflow
- `src/features.py` — cleaning and feature engineering functions
- `src/train.py` — training and evaluation pipeline
- `predict.py` — command-line prediction example
- `app.py` — Streamlit application
- `models/` — saved trained model and metadata
- `reports/` — metrics, charts, and project report
- `screenshots/` — selected project outputs and deployment screenshots

## Limitations

The dataset used in this project is synthetic and is intended for model development and demonstration. Performance on real telecom data may differ. A production deployment would require validation on real customer data, continuous monitoring, privacy controls, model-drift checks, and periodic retraining.

## Author

**Irha Fatima**  
BS Artificial Intelligence  
Bahria University Karachi Campus
