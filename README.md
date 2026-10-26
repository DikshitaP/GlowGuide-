# 🌸 Glow Guide — Personalized Skincare Recommendation System

Glow Guide is a personalized skincare recommendation application designed to help users discover suitable skincare products based on their skin type, skin concerns, budget, and preferences.

The application combines an ingredient knowledge base, product information, and a weighted recommendation algorithm to rank products according to their relevance to the user's needs. It also provides ingredient insights, skincare routine suggestions, and similar-product discovery to make skincare decisions easier.

## ✨ Features

- **Personalized Recommendations:** Recommends skincare products based on the user's skin type and specific concerns.
- **Ingredient Analysis:** Matches relevant skincare ingredients to selected concerns and provides information about their benefits and suitability.
- **Weighted Product Ranking:** Ranks products using ingredient relevance, skin-type compatibility, product quality, category relevance, and value for money.
- **Budget-Based Filtering:** Helps users discover products according to their budget.
- **Explainable Recommendations:** Displays recommendation scores and contributing factors to make product rankings easier to understand.
- **Skincare Routine Builder:** Helps users organise products into routine steps, including cleansing, toning, treatment, moisturising, and optional additional care.
- **Ingredient Compatibility Checks:** Identifies potential ingredient conflicts based on the available ingredient knowledge base.
- **Similar Product Discovery:** Finds products with similar ingredient lists and helps users explore alternative options.
- **User Authentication:** Supports user registration, login, and logout.
- **Interactive Interface:** Provides a clean, skincare-inspired user interface built with Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web application |
| Pandas | Data processing and manipulation |
| NumPy | Numerical operations and array processing |
| SciPy | Scientific computing and similarity calculations |
| Altair | Data visualisation |
| SQLite | Local user account storage |
| PBKDF2-HMAC-SHA256 | Secure password hashing |

## 📂 Project Structure

```text
skincare_app/
│
├── app.py                         # Main Streamlit application
├── auth.py                        # User authentication and password hashing
├── engine.py                      # Recommendation and ranking logic
├── build_data.py                  # Generates portable data files
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── kb.csv                     # Skincare ingredient knowledge base
│   ├── products.csv               # Skincare product catalogue
│   └── arrays.npz                 # Precomputed matrices and scores
│
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
│
└── .gitignore                     # Files excluded from Git
```

## 🚀 Installation and Setup

Follow these steps to run Glow Guide on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub username and repository name.

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a Virtual Environment

A virtual environment keeps the project's dependencies separate from other Python projects.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

From the project's root directory, execute:

```bash
streamlit run app.py
```

Once the application starts, open the local URL displayed in your terminal. It is usually:

```text
http://localhost:8501
```

Create an account using the registration option and log in to explore the application.

## 🧠 How the Recommendation System Works

Glow Guide uses a **weighted, content-based recommendation approach** that evaluates the relationship between a user's skincare requirements and the available product information.

The recommendation process follows these steps:

1. **User Input:** The user selects their skin type, skin concerns, budget, and relevant product preferences.
2. **Ingredient Matching:** The system identifies ingredients associated with the selected skin concerns using the ingredient knowledge base.
3. **Suitability Evaluation:** Products are evaluated according to skin-type compatibility and applicable user preferences.
4. **Product Scoring:** The recommendation engine combines multiple factors to calculate a weighted score for each product.
5. **Ranking:** Products are ranked according to their calculated scores, helping users identify relevant options.
6. **Recommendation Explanation:** The application displays scoring factors to help users understand why a product appears in the results.
7. **Routine and Similarity Support:** Additional functionality helps users organise skincare products into routines and discover products with similar ingredient profiles.

### 📊 Recommendation Scoring

The recommendation engine uses the following default weights:

| Recommendation Factor | Weight |
|---|---:|
| Ingredient Relevance | 55% |
| Skin-Type Compatibility | 15% |
| Product Quality | 15% |
| Category Relevance | 10% |
| Value for Money | 5% |
| **Total** | **100%** |

Ingredient relevance receives the highest weight because the system prioritises how well a product's ingredients match the user's skincare concerns.

The final ranking depends on the available product data, user selections, and the scoring logic implemented in the application. A higher recommendation score indicates a stronger match according to the system; it does not guarantee that a product will be effective or suitable for every individual.

## 📁 Dataset and Knowledge Base

Glow Guide uses two primary datasets and a file containing precomputed data.

### 1. Ingredient Knowledge Base — `kb.csv`

Contains information about skincare ingredients, including:

- Ingredient names and common names
- Ingredient categories and descriptions
- Benefits and associated skin concerns
- Suitable skin types
- Comedogenic ratings and irritation information
- Pregnancy-related safety metadata
- Vegan status and origin
- Typical concentrations and evidence levels
- Ingredient compatibility information

### 2. Product Catalogue — `products.csv`

Contains skincare product information, including:

- Product names and brands
- Product categories and subcategories
- Product prices
- Ingredient lists
- Ratings and number of ratings
- Product links and other available attributes

### 3. Precomputed Data — `arrays.npz`

Stores precomputed matrices and scores used by the recommendation engine.

**Note:** Recommendation quality depends on the completeness and reliability of the underlying data. Missing, outdated, or inaccurate product information may affect the results.

## 🔐 User Authentication

Glow Guide includes a basic authentication system that supports:

- User registration
- Login and logout
- Local account storage using SQLite
- Salted password hashing using PBKDF2-HMAC-SHA256

User account information is stored in a local `users.db` file created when the application runs.

The local database should not be committed to GitHub because it may contain user account information. For production deployment, a persistent database and additional security controls should be considered.

## 🔄 Rebuilding the Data Files (Optional)

The project includes `build_data.py`, which can regenerate the portable data files from the original `skincare_recsys_artifacts.joblib` artifact.

If you have that source artifact, install the additional dependencies:

```bash
pip install joblib scipy scikit-learn
```

Then execute:

```bash
python build_data.py skincare_recsys_artifacts.joblib
```

The generated files are written to the `data/` directory.

This step is optional when the required data files are already included in the repository.

## 🧪 Future Enhancements

Potential improvements include:

- Expanding the ingredient knowledge base and product catalogue.
- Improving handling of missing or incomplete product information.
- Adding automated tests for recommendation accuracy and authentication.
- Introducing a persistent database for deployed applications.
- Updating product prices and availability more regularly.
- Collecting user feedback to evaluate recommendation relevance.
- Improving accessibility and mobile responsiveness.
- Adding more detailed explanations of ingredient suitability and compatibility.

## ⚠️ Disclaimer

Glow Guide is an educational skincare recommendation tool and is not intended to diagnose, treat, or prevent medical conditions.

Recommendations are generated from available ingredient and product data. They do not guarantee clinical effectiveness, personal compatibility, or safety. Individual skin responses vary, and product information may be incomplete or outdated.

Users should review product labels, consider their individual sensitivities, and consult a qualified dermatologist or healthcare professional for persistent skin concerns or questions about ingredient safety.

## 👩‍💻 Project Overview

**Glow Guide — Personalized Skincare Recommendation System**

Built with Python, Streamlit, and a weighted recommendation algorithm to make skincare product discovery more personalised, transparent, and accessible.

---

⭐ If you find this project interesting, consider giving the repository a star!