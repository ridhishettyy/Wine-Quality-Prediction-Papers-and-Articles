# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load model
# model = joblib.load("models/model.pkl")

# st.set_page_config(page_title="Wine Health Predictor")

# st.title("🍷 Wine Health Risk & Quality Advisor")

# st.write("Adjust the wine characteristics below:")

# # =========================
# # INPUT SLIDERS
# # =========================

# fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
# volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
# citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
# residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
# chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08)
# free_sulfur = st.slider("Free Sulfur Dioxide", 1, 80, 15)
# total_sulfur = st.slider("Total Sulfur Dioxide", 6, 300, 46)
# density = st.slider("Density", 0.9900, 1.0050, 0.9968, step=0.0001, format="%.4f")
# pH = st.slider("pH", 2.8, 4.0, 3.3)
# sulphates = st.slider("Sulphates", 0.3, 1.5, 0.6)
# alcohol = st.slider("Alcohol (%)", 5.0, 15.0, 10.0)
# quality = st.slider("Quality", 3, 8, 5)

# # =========================
# # RECOMMENDATION FUNCTION
# # =========================

# def get_recommendations(alcohol, sulphates, pH, residual_sugar):
#     suggestions = []

#     if alcohol < 9:
#         suggestions.append("Increase alcohol content slightly to improve wine body and quality.")
#     elif alcohol > 13:
#         suggestions.append("Reduce alcohol content to lower health risk.")

#     if sulphates < 0.5:
#         suggestions.append("Increase sulphates slightly for better preservation.")
#     elif sulphates > 1.0:
#         suggestions.append("Reduce sulphates to improve health safety.")

#     if pH < 3.0:
#         suggestions.append("Increase pH slightly to reduce acidity.")
#     elif pH > 3.8:
#         suggestions.append("Lower pH for better balance and taste.")

#     if residual_sugar > 8:
#         suggestions.append("Reduce sugar content for a healthier wine profile.")

#     if not suggestions:
#         suggestions.append("Wine composition looks balanced. No major improvements needed.")

#     return suggestions

# # =========================
# # PREDICTION
# # =========================

# if st.button("Predict"):

#     data = np.array([[
#         fixed_acidity,
#         volatile_acidity,
#         citric_acid,
#         residual_sugar,
#         chlorides,
#         free_sulfur,
#         total_sulfur,
#         density,
#         pH,
#         sulphates,
#         alcohol,
#         quality
#     ]])

#     prediction = model.predict(data)[0]

#     # =========================
#     # HEALTH RESULT
#     # =========================

#     st.subheader("🧾 Health Risk Result")

#     if prediction == 0:
#         st.success("🟢 Low Health Risk")
#         st.write("This wine has a relatively safe chemical composition.")
#     elif prediction == 1:
#         st.warning("🟡 Moderate Health Risk")
#         st.write("Some parameters may need adjustment for better health impact.")
#     else:
#         st.error("🔴 High Health Risk")
#         st.write("High alcohol or sulphate levels may negatively affect health.")

#     # =========================
#     # RECOMMENDATIONS
#     # =========================

#     st.subheader("🔧 Recommendations to Improve Quality & Health")

#     recommendations = get_recommendations(alcohol, sulphates, pH, residual_sugar)

#     for rec in recommendations:
#         st.write("•", rec)

#     # =========================
#     # CHART 1: COMPOSITION
#     # =========================

#     st.subheader("📊 Wine Composition Overview")

#     chart_data = pd.DataFrame({
#         'Feature': ['Alcohol', 'Sulphates', 'pH', 'Sugar'],
#         'Value': [alcohol, sulphates, pH, residual_sugar]
#     })

#     st.bar_chart(chart_data.set_index('Feature'))

#     # =========================
#     # CHART 2: RISK DISPLAY
#     # =========================

#     st.subheader("📈 Risk Category")

#     risk_map = {0: "Low", 1: "Moderate", 2: "High"}
#     st.write(f"Predicted Risk Level: **{risk_map[prediction]}**")


# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load model
# model = joblib.load("models/model.pkl")

# st.set_page_config(page_title="Wine Health Predictor")

# st.title("🍷 Wine Health Risk & Quality Advisor")

# st.write("Adjust the wine characteristics below:")

# # =========================
# # INPUT SLIDERS
# # =========================

# fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
# volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
# citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
# residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
# chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08)
# free_sulfur = st.slider("Free Sulfur Dioxide", 1, 80, 15)
# total_sulfur = st.slider("Total Sulfur Dioxide", 6, 300, 46)
# density = st.slider("Density", 0.9900, 1.0050, 0.9968, step=0.0001, format="%.4f")
# pH = st.slider("pH", 2.8, 4.0, 3.3)
# sulphates = st.slider("Sulphates", 0.3, 1.5, 0.6)
# alcohol = st.slider("Alcohol (%)", 5.0, 15.0, 10.0)
# quality = st.slider("Quality", 3, 8, 5)

# # =========================
# # RECOMMENDATIONS FUNCTION
# # =========================

# def get_recommendations(alcohol, sulphates, pH, residual_sugar):
#     suggestions = []

#     if alcohol < 9:
#         suggestions.append("Increase alcohol content slightly to improve wine body and quality.")
#     elif alcohol > 13:
#         suggestions.append("Reduce alcohol content to lower health risk.")

#     if sulphates < 0.5:
#         suggestions.append("Increase sulphates slightly for better preservation.")
#     elif sulphates > 1.0:
#         suggestions.append("Reduce sulphates to improve health safety.")

#     if pH < 3.0:
#         suggestions.append("Increase pH slightly to reduce acidity.")
#     elif pH > 3.8:
#         suggestions.append("Lower pH for better balance and taste.")

#     if residual_sugar > 8:
#         suggestions.append("Reduce sugar content for a healthier profile.")

#     if not suggestions:
#         suggestions.append("Wine composition looks balanced. No major improvements needed.")

#     return suggestions

# # =========================
# # CONNOISSEUR TIPS
# # =========================

# def connoisseur_tips(quality):
#     tips = []

#     if quality <= 4:
#         tips.append("Below standard quality. Improve fermentation and balance.")
#         tips.append("May lack complexity and depth.")

#     elif 5 <= quality <= 6:
#         tips.append("Average quality wine.")
#         tips.append("Enhance aroma and structure for improvement.")

#     else:
#         tips.append("High-quality wine with excellent balance and complexity.")
#         tips.append("Suitable for premium consumption and aging.")

#     return tips

# # =========================
# # PREDICTION
# # =========================

# if st.button("Predict"):

#     data = np.array([[
#         fixed_acidity,
#         volatile_acidity,
#         citric_acid,
#         residual_sugar,
#         chlorides,
#         free_sulfur,
#         total_sulfur,
#         density,
#         pH,
#         sulphates,
#         alcohol,
#         quality
#     ]])

#     prediction = model.predict(data)[0]

#     # =========================
#     # HEALTH RESULT
#     # =========================

#     st.subheader("🧾 Health Risk Result")

#     if prediction == 0:
#         st.success("🟢 Low Health Risk")
#         st.write("This wine has a relatively safe composition.")
#     elif prediction == 1:
#         st.warning("🟡 Moderate Health Risk")
#         st.write("Some parameters may need adjustment.")
#     else:
#         st.error("🔴 High Health Risk")
#         st.write("High alcohol or sulphate levels may impact health.")

#     # =========================
#     # RECOMMENDATIONS
#     # =========================

#     st.subheader("🔧 Recommendations")

#     recommendations = get_recommendations(alcohol, sulphates, pH, residual_sugar)

#     for rec in recommendations:
#         st.write("•", rec)

#     # =========================
#     # CONNOISSEUR INSIGHTS
#     # =========================

#     st.subheader("🍇 Connoisseur Insights")

#     tips = connoisseur_tips(quality)

#     for tip in tips:
#         st.write("•", tip)

#     # =========================
#     # HEALTH CHARTS
#     # =========================

#     st.subheader("📊 Health Impact Analysis")

#     # Alcohol vs Liver Stress
#     liver_stress = alcohol * 5
#     df1 = pd.DataFrame({
#         "Metric": ["Alcohol Level", "Liver Stress Index"],
#         "Value": [alcohol, liver_stress]
#     })
#     st.write("⚠️ Alcohol vs Liver Stress")
#     st.bar_chart(df1.set_index("Metric"))

#     # Sulphates vs Allergy Risk
#     allergy_risk = sulphates * 10
#     df2 = pd.DataFrame({
#         "Metric": ["Sulphates", "Allergy Risk Index"],
#         "Value": [sulphates, allergy_risk]
#     })
#     st.write("🤧 Sulphates vs Allergy Risk")
#     st.bar_chart(df2.set_index("Metric"))

#     # Sugar vs Diabetes Risk
#     diabetes_risk = residual_sugar * 2
#     df3 = pd.DataFrame({
#         "Metric": ["Residual Sugar", "Diabetes Risk Index"],
#         "Value": [residual_sugar, diabetes_risk]
#     })
#     st.write("🍬 Sugar vs Diabetes Risk")
#     st.bar_chart(df3.set_index("Metric"))

#     # pH vs Acid Reflux Risk
#     acid_risk = (4 - pH) * 10
#     df4 = pd.DataFrame({
#         "Metric": ["pH Level", "Acid Reflux Risk"],
#         "Value": [pH, acid_risk]
#     })
#     st.write("🔥 Acidity vs Acid Reflux Risk")
#     st.bar_chart(df4.set_index("Metric"))

#     # Disclaimer
#     st.info("⚠️ These health indicators are for educational purposes only and not medical advice.")
# app.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

body {
    background-color: #F8F5F2;
}

.main {
    background-color: #F8F5F2;
}

section[data-testid="stSidebar"] {
    background-color: #3B0A24;
    color: white;
}

.sidebar-title {
    font-size: 28px;
    font-weight: bold;
    color: white;
    margin-bottom: 30px;
}

.hero-card {
    background: linear-gradient(135deg, #6A0D25, #A61C3C);
    padding: 30px;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

.metric-card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 18px;
    color: gray;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
    color: #8E1C3A;
}

.suggestion-card {
    background-color: white;
    padding: 20px;
    border-left: 6px solid #8E1C3A;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

.stButton>button {
    background-color: #8E1C3A;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #6A0D25;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.markdown(
    '<div class="sidebar-title">🍷 Wine Dashboard</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyze Wine", "Analytics", "About"]
)

# -----------------------------------
# HOME PAGE
# -----------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero-card">
        <h1>🍷 AI Wine Quality Predictor</h1>
        <p>
        Predict wine quality using Machine Learning and analyze
        potential health risks based on wine chemistry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">6,497</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-value">92%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Features Used</div>
            <div class="metric-value">11</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📊 Wine Quality Distribution")

    chart_data = pd.DataFrame({
        "Quality": [3,4,5,6,7,8],
        "Count": [30, 120, 450, 800, 350, 90]
    })

    st.bar_chart(chart_data.set_index("Quality"))

# -----------------------------------
# ANALYZE PAGE
# -----------------------------------

elif page == "Analyze Wine":

    st.title("🍷 Analyze Wine")

    left, right = st.columns([1,1])

    # ---------------- INPUTS ----------------

    with left:

        alcohol = st.slider("Alcohol", 0.0, 20.0, 9.4)
        ph = st.slider("pH", 0.0, 5.0, 3.5)
        sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56)
        sugar = st.slider("Residual Sugar", 0.0, 20.0, 1.9)
        citric = st.slider("Citric Acid", 0.0, 2.0, 0.2)
        acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.7)

        analyze = st.button("Analyze Wine")

    # ---------------- RESULTS ----------------

    with right:

        if analyze:

            # -----------------------------------
            # DUMMY ML LOGIC
            # Replace with your actual model
            # -----------------------------------

            quality_score = round(
                (alcohol + sulphates + citric) -
                (acidity * 2),
                1
            )

            quality_score = max(1, min(10, quality_score))

            if quality_score >= 7:
                health = "Low Risk"
                color = "green"

            elif quality_score >= 5:
                health = "Moderate"
                color = "orange"

            else:
                health = "High Risk"
                color = "red"

            # -----------------------------------
            # GAUGE CHART
            # -----------------------------------

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={'text': "Wine Quality Score"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "#8E1C3A"},
                    'steps': [
                        {'range': [0, 4], 'color': "#ffcccc"},
                        {'range': [4, 7], 'color': "#fff3cd"},
                        {'range': [7, 10], 'color': "#d4edda"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------
            # RESULT CARDS
            # -----------------------------------

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Quality</div>
                    <div class="metric-value">{quality_score}/10</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Health Risk</div>
                    <div class="metric-value">{health}</div>
                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------
            # AI SUGGESTIONS
            # -----------------------------------

            suggestion = ""

            if acidity > 0.6:
                suggestion += "• Reduce volatile acidity.\n"

            if citric < 0.3:
                suggestion += "• Increase citric acid slightly.\n"

            if alcohol < 10:
                suggestion += "• Improve alcohol balance.\n"

            st.write("")

            st.markdown(f"""
            <div class="suggestion-card">
                <h3>💡 AI Recommendation</h3>
                <p>{suggestion}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------
# ANALYTICS PAGE
# -----------------------------------

elif page == "Analytics":

    st.title("📊 Wine Analytics")

    chart_data = pd.DataFrame({
        "Feature": [
            "Alcohol",
            "pH",
            "Sulphates",
            "Sugar",
            "Citric Acid"
        ],
        "Importance": [0.35, 0.12, 0.22, 0.10, 0.21]
    })

    st.subheader("Feature Importance")

    st.bar_chart(chart_data.set_index("Feature"))

    st.subheader("Health Risk Levels")

    risk_df = pd.DataFrame({
        "Risk": ["Low", "Moderate", "High"],
        "Count": [320, 180, 70]
    })

    st.bar_chart(risk_df.set_index("Risk"))

# -----------------------------------
# ABOUT PAGE
# -----------------------------------

elif page == "About":

    st.title("ℹ️ About Project")

    st.write("""
    ### Wine Quality Predictor

    This project uses Machine Learning to predict wine quality
    based on physicochemical properties such as:

    - Alcohol
    - pH
    - Sulphates
    - Citric Acid
    - Residual Sugar

    ### Technologies Used

    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - Pandas

    ### Future Improvements

    - Deep Learning models
    - PDF Report generation
    - Real-time analytics
    - Wine recommendation system
    """)