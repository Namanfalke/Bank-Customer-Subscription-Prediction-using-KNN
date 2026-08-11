import pandas as pd
import streamlit as st
import joblib

knn=joblib.load("model/KNN_model.pkl")
ss=joblib.load("model/StandardScalar.pkl")

st.set_page_config(page_title="My App", layout='wide')
st.title("My App")

# Age	Annual_Income	Credit_Score	Account_Balance	Campaign_Count
# 	Months_With_Bank	Labelled_Education	Labelled_Housing_Loan
# 	Labelled_Personal_Loan Labelled_Contacted_Before


age=st.number_input('Age', min_value=18,max_value=100)
annual_Inc= st.number_input('Annual_Income', min_value=0 )
Cre_Sco= st.number_input('Credit_Score')
acc_bal= st.number_input('Account_Balance', min_value=0)
camp_coun= st.number_input('Campaign_Count')
months_with_bank= st.number_input('Months_With_Bank')
Education=st.number_input("Education for non_graduate == 1, graduate == 0",min_value=0,max_value=1)
Housing_Loan= st.number_input("Labelled_Housing_Loan")
Personal_Loan= st.number_input('Labelled_Personal_Loan')
Contacted_Before= st.number_input('Labelled_Contacted_Before')

input_data=[age, annual_Inc, Cre_Sco, acc_bal, camp_coun, months_with_bank, Education, Housing_Loan, Personal_Loan, Contacted_Before]

if st.button('Submit'):
    scaled_input = ss.transform([input_data])
    # st.write(input_data)
    # st.write(scaled_input)
    prediction = knn.predict(scaled_input)
    if prediction == "yes" :
        st.write(prediction, "you are eligible for taking subscriptions of this bank ")
    elif prediction == "no" :
        st.write(prediction, "you are not eligible for taking subscriptions of this bank ")
