
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model and comparison results
model = joblib.load("models/student_performance_model.joblib")
results = pd.read_csv("model_comparison.csv")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Performance Prediction System")
st.write(
    "An AI/ML-based application that predicts student performance "
    "using academic and lifestyle factors."
)

st.divider()

# Sidebar
st.sidebar.header("Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
previous_marks = st.sidebar.slider("Previous Marks (%)", 0, 100, 70)
study_hours = st.sidebar.slider("Daily Study Hours", 0.0, 12.0, 4.0)
assignment_completion = st.sidebar.slider(
    "Assignment Completion (%)", 0, 100, 80
)
sleep_hours = st.sidebar.slider("Daily Sleep Hours", 0.0, 12.0, 7.0)

# Prediction
if st.sidebar.button("Predict Performance"):
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "previous_marks": previous_marks,
        "study_hours": study_hours,
        "assignment_completion": assignment_completion,
        "sleep_hours": sleep_hours
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))

    st.subheader("Prediction Result")

    st.metric("Predicted Final Marks", f"{prediction:.2f}/100")

    if prediction >= 75:
        st.success("Excellent performance! Keep it up.")
    elif prediction >= 50:
        st.info("Average performance. Consistent effort can improve your score.")
    else:
        st.warning("Needs improvement. Focus on attendance and regular study.")

    st.progress(int(prediction))

st.divider()

# Model comparison
st.subheader("Machine Learning Model Comparison")

st.dataframe(results, use_container_width=True)

fig = px.bar(
    results,
    x="Model",
    y="MAE",
    title="Comparison of Models using Mean Absolute Error",
    color="Model"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: This project uses a synthetically generated dataset "
    "for academic demonstration. Predictions are illustrative, "
    "not actual academic assessments."
)
