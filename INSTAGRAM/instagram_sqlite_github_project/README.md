# Instagram Data Pipeline — SQLite

A small Python ETL project that loads Instagram-related CSV files, normalizes column names and selected values, builds a unified interactions fact table, writes the tables to SQLite, and prints basic row-count summaries.

## Repository structure

```text
.
├── instagram_pipeline.py
├── notebooks/
│   └── instagram_pipeline_walkthrough.ipynb
├── instagram details/          # Add source CSV files here; do not commit private data
├── charts/
│   ├── pipeline_components.png
│   └── interaction_types.png
├── screenshots/
│   └── sample_output.png       # illustrative template, not real run output
├── docs/
│   └── project_report.docx
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.10+
- pandas
- numpy (optional for the original workflow; current script does not require it)

Install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Add data and run

Put the expected CSV files in `instagram details/`. The pipeline expects filename stems:
`users.csv`, `photos.csv`, `tags.csv`, `comments.csv`, `likes.csv`, and `follows.csv`.
Column names should correspond to the source schema used by your export.

```bash
python instagram_pipeline.py
```

The SQLite database `insta_lite.db` is created beside the script. The script prints counts for users, photos, distinct tags, likes, comments, and follows.

## Notebook

Open `notebooks/instagram_pipeline_walkthrough.ipynb` in Jupyter or upload it to Google Colab. The notebook is a walkthrough/template; it uses the same local CSV folder convention. For a live notebook URL, push this repository to GitHub and use the GitHub/Colab URL format described in `docs/project_report.docx`.

## Important implementation notes

- This version fails clearly if required files are missing instead of silently swallowing every loading error.
- `DataFrame.to_sql(..., if_exists="replace")` recreates tables from DataFrame columns and does **not** preserve the primary-key/foreign-key constraints from separate `CREATE TABLE` statements. If database-enforced constraints are required, create tables explicitly and insert rows rather than replacing the tables.
- The original script uses a broad `except: pass`, which can hide malformed CSVs or path problems.
- Date parsing with `errors="coerce"` converts unparseable values to `NaT`; inspect parse-failure rates before relying on dates for analysis.
- Row-based IDs are reproducible only when row order remains stable. Prefer source IDs when available.
- Do not commit personal or sensitive exports. Review the data license and remove identifying fields before publishing.

## Push to GitHub

Create an empty repository on GitHub, then from this folder run:

```bash
git init
git add README.md instagram_pipeline.py notebooks charts screenshots docs requirements.txt .gitignore
git commit -m "Add Instagram CSV to SQLite analytics pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL. If Git reports an existing remote, inspect it with `git remote -v` rather than adding a duplicate.
