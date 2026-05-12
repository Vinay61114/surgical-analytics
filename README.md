# Surgical Analytics Dashboard

Anterior lumbar approach complication risk analysis — 331 patients, AL-LIF Retro.

## What this app does

Five-panel interactive dashboard:
1. **Demographics** — age, sex, BMI, ASA, comorbidity prevalence
2. **Complications** — 30-day outcome rates, breakdown by ASA and BMI
3. **Exposure time** — scatter, binned rates, prior surgery violin plot
4. **Model** — SHAP importance, three-model regression table, ROC curve
5. **Risk calculator** — live probability estimate with patient inputs

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Place your data file in the same folder as app.py
cp /path/to/data_deidentified.csv .

# Run
streamlit run app.py
# Open http://localhost:8501
```

## Deploy to Streamlit Cloud (step by step)

### Step 1 — Create a GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Give it a name: `streamlit-surgical`
4. Set expiration: 90 days (or no expiration)
5. Check the box: **repo** (full repository access)
6. Click **Generate token**
7. **Copy the token immediately** — you will not see it again
8. Save it somewhere safe (e.g. your password manager)

This token is your password substitute. Your GitHub username is `Jay`.

### Step 2 — Create a GitHub repository

1. Go to https://github.com/new
2. Repository name: `surgical-analytics`
3. Set to **Private** (important — keeps your data references private)
4. Do NOT initialise with README (you already have one)
5. Click **Create repository**

### Step 3 — Push this code to GitHub

Open a terminal in the folder containing app.py and run:

```bash
# Initialise git (first time only)
git init
git add app.py requirements.txt .streamlit/config.toml .gitignore README.md

# IMPORTANT: do NOT add the CSV file — it should never go to GitHub
# Confirm .gitignore is excluding it:
git status  # data_deidentified.csv should NOT appear

# Commit
git commit -m "Initial surgical analytics dashboard"

# Connect to your GitHub repo (replace Jay with your actual username)
git remote add origin https://github.com/Jay/surgical-analytics.git

# Push (use your token as the password when prompted)
git push -u origin main
# Username: Jay
# Password: [paste your Personal Access Token here]
```

### Step 4 — Upload data to Streamlit Cloud securely

The CSV must not be on GitHub. Upload it through Streamlit Cloud secrets instead.

**Option A (recommended): Upload via Streamlit Cloud file uploader**
1. Go to https://share.streamlit.io
2. After deploying, go to your app settings
3. Use the Secrets section to store a base64-encoded version of your CSV

**Option B (simplest for a private repo): Include the CSV directly**
Since the repo is private and only you have access, you can include the
de-identified CSV for a private repository. To do this:

```bash
# Remove data_deidentified.csv from .gitignore
# Edit .gitignore and delete or comment out the *.csv line

# Then add and push the data
git add data_deidentified.csv
git commit -m "Add de-identified dataset"
git push
```

⚠️ Only do this for a **private** repository with de-identified data.

### Step 5 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account (username: Jay)
3. Click **New app**
4. Select repository: `Jay/surgical-analytics`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**

Streamlit Cloud will install requirements.txt automatically and launch the app.
You will get a public URL like: `https://jay-surgical-analytics.streamlit.app`

### Step 6 — Access control (optional)

To restrict who can see the app:
1. In Streamlit Cloud, go to your app → Settings → Sharing
2. Set to **Only specific people can view this app**
3. Add email addresses of authorised viewers

---

## File structure

```
surgical-analytics/
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← Python dependencies
├── .gitignore                ← Keeps CSV and secrets out of GitHub
├── README.md                 ← This file
├── .streamlit/
│   └── config.toml           ← Theme and server settings
└── data_deidentified.csv     ← Dataset (private repo only)
```

## Model details

- **Algorithm:** Logistic regression (primary), XGBoost (SHAP/explainability)
- **AUC:** 0.693 (XGBoost), 0.691 (logistic), 0.739 (M2 sequential model)
- **n:** 331 patients, 52 complications (15.7%)
- **Key finding:** Exposure time is not independently significant (OR 1.004, p=0.92)
  after controlling for age (OR 1.049, p<0.001) and surgical time (OR 1.015, p=0.02)

## Data notes

- Source: Prospective surgical registry, single centre
- Approach: AL-LIF retroperitoneal (all 331 cases)
- De-identified: Patient name, MR#, DOB, surgeon names removed
- Cleaning: 7-step pipeline in `clean_data.py`
