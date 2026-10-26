# Glow Guide: Streamlit frontend for the Skincare Recommender v2

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
`data/` is already built from your `skincare_recsys_artifacts.joblib`, so it works straight away.

## What's in the app
- **Products**: ranked cards with price, rating and key actives
- **Why these?**: stacked chart of what makes up each product's score, plus a per-product breakdown
- **Ingredients**: what to look for on the label, with evidence level and what not to mix
- **Routine**: cleanser / serum / moisturiser with conflict warnings and total cost
- **Find similar**: look-alike (and cheaper) products by ingredient list

## Login / logout
The app opens on a login page with a "Create account" tab. Accounts are stored in `users.db` (created automatically next to `app.py`)
with salted PBKDF2 password hashes, never plain text. The sidebar shows who is signed in and has a **Log out** button,
which leads to a "You've been signed out" page.
Note: on Streamlit Community Cloud the disk resets when the app restarts, so accounts made there can disappear.
For a permanent user list, host the app somewhere with a persistent disk, or point `AUTH_DB` at a database file on one.

## If you retrain the model
```
pip install scikit-learn joblib
python build_data.py skincare_recsys_artifacts.joblib
```
This rewrites `data/` (plain CSV and NumPy files, so no version clashes between machines).

## Files
- `app.py`: the web UI
- `auth.py`: sign-up / login (SQLite + hashed passwords)
- `engine.py`: the notebook's Section 6 scoring engine (same logic and weights)
- `build_data.py`: converts the notebook's joblib into `data/`
- `data/`: kb.csv (ingredients), products.csv (1,926 products), arrays.npz (matrices and ML scores)
- `.streamlit/config.toml`: theme colours

## Deploy
Push this folder to GitHub, then create an app on Streamlit Community Cloud with `app.py` as the entry file.
