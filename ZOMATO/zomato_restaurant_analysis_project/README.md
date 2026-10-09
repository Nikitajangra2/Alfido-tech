# Zomato Restaurant Data Analysis

## Executive Summary
This project explores restaurant data using data cleaning, exploratory data analysis (EDA), feature engineering, supervised learning, regression diagnostics, and unsupervised clustering. It converts restaurant price-for-two into a binary `expensive` label based on the dataset median, compares four classifiers, examines regression RMSE, and explores four K-Means clusters with t-SNE visualization.

**Dataset:** `zomato.csv` (not included in this package).

## Main Components
- Data cleaning and missing-value handling
- Distribution, category-count, pairplot, and boxplot visualizations
- One-hot encoding and feature scaling
- Classification metrics: accuracy, precision, recall, F1
- Regression RMSE diagnostics
- K-Means elbow chart and cluster summaries
- t-SNE visualization of sampled training data

## Files
- `zomato_restaurant_analysis.ipynb` — Jupyter Notebook
- `zomato_restaurant_analysis.py` — Python script
- `REPORT.docx` — executive summary and project report
- `requirements.txt` — dependencies
- `project_workflow.png` — workflow illustration
- `notebook_preview.png` — generated preview, not an executed screenshot

## Run
1. Put `zomato.csv` in the project folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `jupyter notebook zomato_restaurant_analysis.ipynb`

## GitHub Push
```bash
git init
git add .
git commit -m "Add Zomato restaurant analysis project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/zomato-restaurant-analysis.git
git push -u origin main
```

Notebook URL after publishing:
`https://github.com/YOUR_USERNAME/zomato-restaurant-analysis/blob/main/zomato_restaurant_analysis.ipynb`

## Methodology Notes
- The `expensive` target is derived from median cost for two people.
- The regression section uses this same binary target, not restaurant ratings; RMSE should not be described as rating-prediction performance.
- For strict evaluation, preprocessing should be fitted inside a scikit-learn Pipeline within each training fold to avoid leakage.
- t-SNE is a visualization technique and should not be treated as a formal cluster-quality score.
- Include `zomato.csv` in GitHub only if its license/terms allow redistribution.
