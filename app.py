import streamlit as st

st.set_page_config(page_title='Maternal Risk Prediction App', layout='wide')
st.title('Maternal Risk Prediction App')

st.write('This App predict probability of pregnant mother to either be having high or low risk of pregnant complications')
st.caption('Developed by Dr. Elisha Magobo')

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input('Age of the Pregnant mother',18, 70, 30)

    Systolic_BP = st.number_input('Systolic Blood Pressure(mmHg)',90, 160, 120)

    Diastolic_BP = st.number_input('Diastolic Blood Pressure(mmHg)',60, 100, 90)

    Blood_Sugar = st.number_input('Blood Sugar Level(mg/dL)',5.0, 28.0, 15.6)

    BMI = st.number_input('Body Mass Index',10.0, 25.0, 18.0)
with col2:
    Previous_Complications = st.selectbox('History of previous Complications',[0, 1],
                                          format_func=lambda x: "Yes" if x == 1 else "No")

    Preexisting_Diabetes = st.selectbox('Preexisting Diabetes',[0, 1],
                                        format_func=lambda x: "Yes" if x == 1 else "No")

    Gestational_Diabetes = st.selectbox('Gestational Diabetes',[0, 1],
                                        format_func=lambda x: "Yes" if x == 1 else "No")

    Mental_Health = st.selectbox('Mental Health Condition',[0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No")

    Heart_Rate = st.number_input('Maximum Heart Rate',60.0, 220.0, 135.0)

st.button('Predict')