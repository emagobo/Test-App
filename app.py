import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# MODELS
bundle = joblib.load("maternal_risk_pipeline.pkl")

best_model = bundle["model"]
scaler = bundle["scaler"]
LabelEncoder = bundle["encoder"]
features = bundle["features"]


st.set_page_config(page_title='Maternal Risk Prediction App', layout='wide', page_icon='')
st.title('Maternal Risk Prediction App')
st.image("dataset_cover.png", use_container_width=True)
st.markdown(
    "<p style='font-size:12px; font-weight:bold;margin-top:0px; margin-bottom:10px;'>Better Decisions for Better Care—At Every Point of the Health Journey. </p>",
    unsafe_allow_html=True
)


st.write("Enter and Select patient clinical parameters below:")

with st.sidebar:
    st.header('Welcome to Maternal Risk App')
    st.image("images.jpg", use_container_width=True)
    st.markdown('**About**')
    st.markdown(
        """
        <div style='text-align: justify; padding: 10px;'>
        This App predicts probability of maternal risk during pregnancy using  routine clinical data, 
        providing a continuous framework for identification across pregnancy and early life.
        
        
        It supports clinicians with rapid, point-of-care, data-driven risk assessment based on inputs
        such as blood pressure, blood sugar, temperature, and obstetric history, classifying patients
        into low- or high-risk categories to guide triage and clinical decisions without disrupting workflows.
        
        By standardizing assessment, the tool reduces variability, enables earlier detection of high-risk cases,
        and highlights key risk drivers to enhance clinical interpretation. It ultimately promotes consistent care,
        timely intervention, and more efficient use of limited healthcare resources.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("elisha.png", width=90)
    st.markdown(
        "<p style='font-size:11px; font-weight:bold;margin-top:0px; margin-bottom:0px;'>®Developed by Dr. Elisha Magobo|@2026 |"
        " Machine Learning | Maternal Project </p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:11px; font-weight:bold; margin-top:0px; margin-bottom:0px;'>Medical Doctor | Public Health Specialist | Data Scientist | Clinical Researcher </p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:11px; font-weight:bold;margin-top:0px; margin-bottom:0px;'>Email: eamagobo@gmail.com | +255 744 650 037 </p>",
        unsafe_allow_html=True
    )

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

#=================================
# FEATURE BUILDER
#=================================
def build_maternal_features():
    return pd.DataFrame([{
            "Age": Age,
            "Systolic_BP": Systolic_BP,
            "Diastolic_BP": Diastolic_BP,
            "Blood_Sugar": Blood_Sugar,
            "BMI": BMI,
            "Previous_Complications": Previous_Complications,
            "Preexisting_Diabetes": Preexisting_Diabetes,
            "Gestational_Diabetes": Gestational_Diabetes,
            "Mental_Health": Mental_Health,
            "Heart_Rate": Heart_Rate
        }])

# ====================================
# PREDICTION BUTTON + RESULTS
# ====================================


if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probabilities = None

if st.button("Predict Risk", key="predict_maternal"):

    try:
        # Build input features
        input_df = build_maternal_features()

        # Ensure column order matches training
        input_df = input_df[features]

        # Apply scaler if used
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]

        # Store in session
        st.session_state.prediction = prediction
        st.session_state.probabilities = probability

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ====================================
# DISPLAY RESULTS
# ====================================

st.subheader("Prediction Results")

if st.session_state.prediction is not None:

    pred = st.session_state.prediction
    probs = st.session_state.probabilities

    if pred == 0:
        st.success("✅ Low Risk — Continue routine ANC follow-up")
    else:
        st.warning("⚠️ High Risk — Consider referral for specialist care")

    # =========================
    # PROBABILITIES
    # =========================
    st.subheader("Prediction Probabilities")

    st.write(f"Low Risk: {probs[0]*100:.2f}%")
    st.write(f"High Risk: {probs[1]*100:.2f}%")




# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("Key Risk Drivers")

try:
    # Check if model supports feature importance
    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        # Plot
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # Top drivers for THIS prediction (simple approximation)
        st.markdown("**Top Contributing Factors:**")
        top_features = importance_df.head(3)["Feature"].values

        for f in top_features:
            value = input_df[f].values[0]
            st.write(f"- {f}: {value}")

    else:
        st.info("Feature importance not available for this model type.")

except Exception as e:
    st.warning(f"Could not compute feature importance: {e}")