# BeeNeural Submission Checklist

## Before GitHub Upload
- [ ] Extract the ZIP and confirm all folders are present.
- [ ] Run `pip install -r requirements.txt`.
- [ ] Run `python predict.py` successfully.
- [ ] Launch `streamlit run app.py` and take one screenshot of the working web app.
- [ ] Optionally rerun `src/train.py` if you want fresh local artifacts.

## GitHub Repository
- [ ] Create a new public repository, e.g. `telecom-customer-churn-prediction`.
- [ ] Upload/push all project files.
- [ ] Confirm `README.md` renders correctly.
- [ ] Confirm `reports/Churn_Analysis_Report.pdf` opens from GitHub.
- [ ] Confirm evaluation images render in `reports/figures/`.
- [ ] Do not upload `.venv/`.

## BeeNeural Task Submission
- [ ] Paste GitHub repository URL.
- [ ] Upload `Churn_Analysis_Report.pdf`.
- [ ] Upload evaluation screenshots.
- [ ] Upload Streamlit app screenshot.
- [ ] Upload ZIP if an attachment slot is available.
- [ ] In submission notes, mention: model comparison, feature engineering, ROC-AUC/PR-AUC evaluation, threshold tuning, and Streamlit deployment.

## Rubric Coverage
- [x] Model Accuracy: baseline comparison + tuning + saved best model.
- [x] Feature Engineering: 11 added business features.
- [x] Model Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, curves, threshold study, feature importance.
- [x] Deployment: Streamlit app + CLI predictor + Docker configuration.
