# Loan Approval Prediction

## Executive Summary

This project predicts loan approval outcomes using applicant information and compares five classification algorithms. The workflow covers data inspection, missing-value treatment, categorical encoding, 5-fold cross-validation, and model-performance visualization.

> **Important:** Accuracy values are generated when the notebook is executed with `loan_prediction.csv`. No performance figures are hard-coded in this repository.

## Models Compared

1. Gradient Boosting
2. Random Forest
3. Decision Tree
4. K-Nearest Neighbor
5. Linear SVM

## Project Structure

- `loan_approval_prediction.ipynb` — main Jupyter notebook
- `loan_approval_prediction.py` — Python script version
- `requirements.txt` — Python dependencies
- `loan_prediction.csv` — input dataset (add locally if permitted)
- `model_comparison.png` — generated after notebook execution
- `REPORT.docx` — project report

## Run Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd loan-approval-prediction

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
jupyter notebook loan_approval_prediction.ipynb
```

Place `loan_prediction.csv` in the repository root before running the notebook.

## GitHub Push

```bash
git init
git add .
git commit -m "Add loan approval prediction project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Recommended GitHub Notebook Link

After pushing, the notebook can be opened directly from:

`https://github.com/<YOUR_USERNAME>/loan-approval-prediction/blob/main/loan_approval_prediction.ipynb`

Replace `<YOUR_USERNAME>` with your GitHub username.

## Notes / Limitations

- Mean imputation is used for `LoanAmount`; mode imputation is used for selected categorical/discrete fields.
- KNN and Linear SVM can be sensitive to feature scale; a production-quality version should use a preprocessing pipeline with scaling.
- The current encoding treats `Property_Area` as numeric labels. One-hot encoding would generally be preferable for nominal categories.
- For a real lending decision system, additional evaluation such as precision, recall, ROC-AUC, confusion matrix, calibration, fairness analysis, and cost-sensitive evaluation should be considered.
