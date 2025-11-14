
# 📊 **Loan Default Prediction System**

### *A Machine Learning Project by Virendra Mahajan*

This project predicts whether a loan applicant is likely to **default** based on their financial and personal information.
It includes **data preprocessing, EDA, model comparison, evaluation**, and a fully interactive **Streamlit web app** for live predictions.

---

## 🚀 **Project Features**

### 🔹 **1. Data Preprocessing**

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Removing identifier columns
* Saving a clean processed dataset

### 🔹 **2. Model Training**

Trained multiple models:

* Logistic Regression
* Random Forest
* (Optional) XGBoost

Metrics used:

* Accuracy
* F1-Score
* ROC-AUC

The best model was saved as:

```
src/models/model.joblib
```

### 🔹 **3. Model Evaluation**

Generated:

* Confusion matrix
* Classification report
* ROC-AUC score
* Feature importance plot
* Model comparison table

### 🔹 **4. Interactive Streamlit Web App**

Allows user inputs:

* Age
* Income
* Loan Amount
* Term
* Credit Score
* Employment Years
* Home Ownership
* Loan Purpose

The app predicts **Default** or **No Default** in real-time.

---

# 📁 **Project Structure**

```
Loan_Default_Prediction/
│── data/
│   ├── loan_data_sample.csv
│   └── processed/
│
│── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── feature_importance.py
│   └── app.py   ← Streamlit App
│
│── reports/
│   ├── Final_Report_VirendraMahajan.pdf
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
│── src/figures/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
│── requirements.txt
│── README.md
```

---

# 🧠 **How to Run Locally**

### 1️⃣ Create & activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run preprocessing

```bash
python src/preprocess.py
```

### 4️⃣ Train the model

```bash
python src/train_model.py --input data/loan_data_sample.csv --out src/models/model.joblib
```

### 5️⃣ Evaluate model

```bash
python src/evaluate.py src/models/model.joblib data/loan_data_sample.csv
```

### 6️⃣ Launch Streamlit App

```bash
streamlit run src/app.py
```

---

# 🌐 **🚀 Deploy the Streamlit App (Free Hosting)**

Follow these steps to deploy online:

---

## ✔ Step 1 — Push your full project to GitHub

Make sure:

* your repo contains: `src/app.py`
* your repo contains: `requirements.txt`
* your repo is public

---

## ✔ Step 2 — Go to Streamlit Cloud

Visit:

👉 https://loandefaultprediction-sygptumgivnwk2endmvsui.streamlit.app/

Log in with GitHub.

---

## ✔ Step 3 — Deploy

Click:

**"New app" → Select your GitHub repo**
Then set:

| Setting    | Value                            |
| ---------- | -------------------------------- |
| Repository | `Kmennn/Loan_Default_Prediction` |
| Branch     | `main`                           |
| App file   | `src/app.py`                     |

Click **Deploy** 🎉

Your app will be LIVE on a public link like:

```
https://loan-default-prediction.streamlit.app
```

---

# 🎓 **Project Report (PDF)**

The final PDF includes:

* Introduction
* Objective
* Methodology
* EDA
* Model details
* Results & observations
* Confusion matrix
* Feature importance
* Conclusion
* Links to dataset & GitHub repo

---

# 🏁 **Conclusion**

This project demonstrates:

* Real ML workflow
* Strong model evaluation
* Best-practice pipeline
* Web deployment
* Clean project architecture

A complete, industry-style **end-to-end data science project**.

---


