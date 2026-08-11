# 🏦 Bank Customer Subscription Prediction using KNN

🤖 Interactive Bank Customer Subscription Prediction Web App built with **Python, Streamlit, and Machine Learning**. Uses the **KNN classification algorithm** and data preprocessing techniques to predict whether a bank customer is likely to subscribe to an offered banking product based on age, job, education, annual income, credit score, account balance, loans, previous contact, campaign count, and banking relationship duration, with an interactive Light/Dark theme UI.

## ▶️ How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/Namanfalke/Bank_Customer_Subscription_KNN.git
```

### 2. Open the Project Folder

```bash
cd Bank_Customer_Subscription_KNN
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

### 7. Open in Browser

After running the command, Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open this URL in your browser to use the application.

## 📁 Project Structure

```text
Bank_Customer_Subscription_KNN/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Dataset/
│   └── Bank_Customer_Subscription_KNN_Dataset.xlsx
│
└── saved model/
    └── KNN_model.pkl
```

> **Note:** Make sure the `saved model` folder and the `KNN_model.pkl` file are present in the repository. The application needs the trained KNN model to make predictions.

## 🎯 Prediction

The application predicts whether a customer is likely to subscribe to the bank's offered product:

```text
Yes → Customer is likely to subscribe
No  → Customer is unlikely to subscribe
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Excel
* KNN Classification
* StandardScaler

## 📊 Features Used

* Age
* Job
* Marital Status
* Education
* Annual Income
* Credit Score
* Account Balance
* Housing Loan
* Personal Loan
* Previous Contact
* Campaign Count
* Months With Bank

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
KNN Model
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit Web App
```
