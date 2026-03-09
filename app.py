import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("loan_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Loan Approval Prediction System")

st.write("Enter applicant details below")

Gender = st.selectbox("Gender",["Male","Female"])
Married = st.selectbox("Married",["Yes","No"])
Dependents = st.selectbox("Dependents",["0","1","2","3+"])
Education = st.selectbox("Education",["Graduate","Not Graduate"])
Self_Employed = st.selectbox("Self Employed",["Yes","No"])

ApplicantIncome = st.number_input("Applicant Income",min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income",min_value=0)

LoanAmount = st.number_input("Loan Amount",min_value=0)
Loan_Amount_Term = st.number_input("Loan Term",min_value=0)

Credit_History = st.selectbox("Credit History",[1,0])

Property_Area = st.selectbox("Property Area",
["Urban","Semiurban","Rural"])

# Encoding (same as training)
gender = 1 if Gender=="Male" else 0
married = 1 if Married=="Yes" else 0
education = 0 if Education=="Graduate" else 1
self_emp = 1 if Self_Employed=="Yes" else 0

if Property_Area=="Urban":
    prop=2
elif Property_Area=="Semiurban":
    prop=1
else:
    prop=0

dep = {"0":0,"1":1,"2":2,"3+":3}[Dependents]

if st.button("Predict"):

    input_data = np.array([[gender,married,dep,education,
                            self_emp,ApplicantIncome,
                            CoapplicantIncome,LoanAmount,
                            Loan_Amount_Term,Credit_History,
                            prop]])

    # Apply scaling
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0]==1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")