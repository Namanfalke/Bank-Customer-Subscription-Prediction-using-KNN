# import pandas as pd
# import streamlit as st
# import joblib

# knn=joblib.load("model/KNN_model.pkl")
# ss=joblib.load("model/StandardScalar.pkl")

# st.set_page_config(page_title="My App", layout='wide')
# st.title("My App")

# # Age	Annual_Income	Credit_Score	Account_Balance	Campaign_Count
# # 	Months_With_Bank	Labelled_Education	Labelled_Housing_Loan
# # 	Labelled_Personal_Loan Labelled_Contacted_Before


# age=st.number_input('Age', min_value=18,max_value=100)
# annual_Inc= st.number_input('Annual_Income', min_value=0 )
# Cre_Sco= st.number_input('Credit_Score')
# acc_bal= st.number_input('Account_Balance', min_value=0)
# camp_coun= st.number_input('Campaign_Count')
# months_with_bank= st.number_input('Months_With_Bank')
# Education=st.number_input("Education for non_graduate == 1, graduate == 0",min_value=0,max_value=1)
# Housing_Loan= st.number_input("Labelled_Housing_Loan")
# Personal_Loan= st.number_input('Labelled_Personal_Loan')
# Contacted_Before= st.number_input('Labelled_Contacted_Before')

# input_data=[age, annual_Inc, Cre_Sco, acc_bal, camp_coun, months_with_bank, Education, Housing_Loan, Personal_Loan, Contacted_Before]

# if st.button('Submit'):
#     scaled_input = ss.transform([input_data])
#     # st.write(input_data)
#     # st.write(scaled_input)
#     prediction = knn.predict(scaled_input)
#     if prediction == "Yes":
#         st.success("✅ Customer is likely to subscribe")
#     else:
#         st.error("❌ Customer is unlikely to subscribe")


# -----------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Subscription Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME SETUP
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


# =========================================================
# THEME COLORS
# =========================================================

if st.session_state.dark_mode:

    background = "#0f172a"
    secondary_background = "#1e293b"
    card_background = "rgba(255,255,255,0.08)"
    text_color = "#ffffff"
    secondary_text = "#cbd5e1"
    border_color = "rgba(255,255,255,0.15)"
    sidebar_background = "#111827"

else:

    background = "#f1f5f9"
    secondary_background = "#ffffff"
    card_background = "rgba(255,255,255,0.9)"
    text_color = "#0f172a"
    secondary_text = "#475569"
    border_color = "#cbd5e1"
    sidebar_background = "#ffffff"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* ==============================
       MAIN APP
       ============================== */

    .stApp {{
        background: linear-gradient(
            135deg,
            {background},
            {secondary_background}
        );

        color: {text_color};
    }}


    /* ==============================
       MAIN TITLE
       ============================== */

    .main-title {{
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: {text_color};
    }}


    .subtitle {{
        text-align: center;
        color: {secondary_text};
        font-size: 18px;
        margin-bottom: 30px;
    }}


    /* ==============================
       SECTION HEADINGS
       ============================== */

    .section-title {{
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
        color: {text_color};
    }}


    /* ==============================
       CARDS
       ============================== */

    .card {{
        background: {card_background};
        border: 1px solid {border_color};
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }}


    /* ==============================
       SUCCESS CARD
       ============================== */

    .success-card {{
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }}


    /* ==============================
       ERROR CARD
       ============================== */

    .danger-card {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }}


    .prediction-text {{
        font-size: 28px;
        font-weight: 800;
    }}


    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {{
        background: {sidebar_background};
    }}


    /* ==============================
       BUTTONS
       ============================== */

    .stButton > button {{
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
    }}


    /* ==============================
       INPUT BOXES
       ============================== */

    div[data-baseweb="input"] {{
        border-radius: 8px;
    }}


    /* ==============================
       METRIC CARDS
       ============================== */

    div[data-testid="stMetric"] {{
        background: {card_background};
        border: 1px solid {border_color};
        padding: 15px;
        border-radius: 12px;
    }}


    /* ==============================
       FOOTER
       ============================== */

    .footer {{
        text-align: center;
        color: {secondary_text};
        padding: 15px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

# BASE_DIR = Path(__file__).resolve().parent

# model_path = BASE_DIR / "model" / "KNN_model.pkl"
# scaler_path = BASE_DIR / "model" / "StandardScalar.pkl"

# knn = joblib.load(model_path)
# ss = joblib.load(scaler_path)

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR.parent / "model" / "KNN_model.pkl"
scaler_path = BASE_DIR.parent / "model" / "StandardScalar.pkl"

knn = joblib.load(model_path)
ss = joblib.load(scaler_path)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    # Theme button
    if st.session_state.dark_mode:

        st.button(
            "☀️ Switch to Light Mode",
            on_click=toggle_theme,
            use_container_width=True
        )

    else:

        st.button(
            "🌙 Switch to Dark Mode",
            on_click=toggle_theme,
            use_container_width=True
        )


    st.divider()


    # Theme status
    if st.session_state.dark_mode:
        st.success("🌙 Dark Mode Active")
    else:
        st.info("☀️ Light Mode Active")


    st.divider()


    # About section
    st.markdown("## 📊 About This App")

    st.write(
        """
        This application uses a **K-Nearest Neighbors (KNN)**
        machine learning model to predict whether a customer
        is likely to subscribe to the service.
        """
    )


    st.divider()


    st.markdown("### 🤖 Machine Learning Model")

    st.info("K-Nearest Neighbors (KNN)")


    st.markdown("### 📌 Input Features")

    st.write(
        """
        • Age  
        • Annual Income  
        • Credit Score  
        • Account Balance  
        • Campaign Count  
        • Months With Bank  
        • Education  
        • Housing Loan  
        • Personal Loan  
        • Contacted Before
        """
    )


    st.divider()

    st.caption(
        "Customer Subscription Prediction System"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '📊 Customer Subscription Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a customer is likely to subscribe '
    'using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '👤 Customer Information'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with col2:

    annual_Inc = st.number_input(
        "💰 Annual Income",
        min_value=0,
        value=50000,
        step=1000
    )


with col3:

    Cre_Sco = st.number_input(
        "💳 Credit Score",
        min_value=0,
        max_value=900,
        value=650,
        step=1
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '💰 Financial Information'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    acc_bal = st.number_input(
        "🏦 Account Balance",
        min_value=0,
        value=10000,
        step=500
    )


with col2:

    camp_coun = st.number_input(
        "📢 Campaign Count",
        min_value=0,
        value=1,
        step=1
    )


with col3:

    months_with_bank = st.number_input(
        "📅 Months With Bank",
        min_value=0,
        value=24,
        step=1
    )


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📋 Customer Profile'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    Education = st.selectbox(
        "🎓 Education",
        options=[0, 1],
        format_func=lambda x:
            "Graduate" if x == 0 else "Non-Graduate"
    )


    Housing_Loan = st.selectbox(
        "🏠 Housing Loan",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    Personal_Loan = st.selectbox(
        "💵 Personal Loan",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


    Contacted_Before = st.selectbox(
        "📞 Contacted Before",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


# =========================================================
# INPUT DATA
# =========================================================

input_data = [
    age,
    annual_Inc,
    Cre_Sco,
    acc_bal,
    camp_coun,
    months_with_bank,
    Education,
    Housing_Loan,
    Personal_Loan,
    Contacted_Before
]


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")


col1, col2, col3 = st.columns([1, 2, 1])


with col2:

    submit = st.button(
        "🔮 Predict Customer Subscription",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submit:

    input_df = pd.DataFrame(
        [input_data],
        columns=[
            "Age",
            "Annual_Income",
            "Credit_Score",
            "Account_Balance",
            "Campaign_Count",
            "Months_With_Bank",
            "Labelled_Education",
            "Labelled_Housing_Loan",
            "Labelled_Personal_Loan",
            "Labelled_Contacted_Before"
        ]
    )


    # Scale input
    scaled_input = ss.transform(input_df)


    # Prediction
    prediction = knn.predict(scaled_input)


    # predict() returns an array
    result = prediction[0]

    result = str(result).lower()


    st.markdown("---")


    # =====================================================
    # RESULT
    # =====================================================

    if result == "yes":

        st.markdown(
            """
            <div class="success-card">

                <div class="prediction-text">
                    ✅ Customer is Likely to Subscribe
                </div>

                <p>
                    The KNN model predicts that this customer
                    is likely to subscribe to the service.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()


    else:

        st.markdown(
            """
            <div class="danger-card">

                <div class="prediction-text">
                    ❌ Customer is Unlikely to Subscribe
                </div>

                <p>
                    The KNN model predicts that this customer
                    is unlikely to subscribe to the service.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📋 Customer Input Summary'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric("Age", age)


    with col2:
        st.metric(
            "Annual Income",
            f"₹{annual_Inc:,}"
        )


    with col3:
        st.metric(
            "Credit Score",
            Cre_Sco
        )


    with col4:
        st.metric(
            "Account Balance",
            f"₹{acc_bal:,}"
        )


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Campaign Count",
            camp_coun
        )


    with col2:
        st.metric(
            "Months With Bank",
            months_with_bank
        )


    with col3:
        st.metric(
            "Housing Loan",
            "Yes" if Housing_Loan == 1 else "No"
        )


    with col4:
        st.metric(
            "Personal Loan",
            "Yes" if Personal_Loan == 1 else "No"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🤖 Powered by KNN Machine Learning | '
    'Customer Subscription Prediction'
    '</div>',
    unsafe_allow_html=True
)


