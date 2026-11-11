
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL AND RESULTS ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/student_performance_model.joblib")

model = load_model()
results = pd.read_csv("model_comparison.csv")
feature_importance = pd.read_csv("feature_importance.csv")

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Prediction System")
st.write(
    "An AI/ML-powered application that predicts student "
    "academic performance using academic and lifestyle factors."
)

st.caption("Predict • Analyze • Improve")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📋 Student Details")
st.sidebar.write("Enter the student's information below.")

attendance = st.sidebar.slider(
    "Attendance (%)", 0, 100, 75
)

previous_marks = st.sidebar.slider(
    "Previous Marks (%)", 0, 100, 70
)

study_hours = st.sidebar.slider(
    "Daily Study Hours", 0.0, 12.0, 4.0, 0.5
)

assignment_completion = st.sidebar.slider(
    "Assignment Completion (%)", 0, 100, 80
)

sleep_hours = st.sidebar.slider(
    "Daily Sleep Hours", 0.0, 12.0, 7.0, 0.5
)

predict_button = st.sidebar.button(
    "🔍 Predict Performance",
    use_container_width=True
)

# ---------------- HELPER FUNCTIONS ----------------
def classify_performance(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"


def generate_recommendations(
    attendance,
    previous_marks,
    study_hours,
    assignment_completion,
    sleep_hours
):
    recommendations = []

    if attendance < 75:
        recommendations.append(
            "📚 Improve attendance by attending classes regularly."
        )

    if previous_marks < 60:
        recommendations.append(
            "📖 Revise previous topics and strengthen basic concepts."
        )

    if study_hours < 3:
        recommendations.append(
            "⏰ Increase daily study time gradually and follow a timetable."
        )

    if assignment_completion < 75:
        recommendations.append(
            "📝 Complete assignments on time and practice regularly."
        )

    if sleep_hours < 6:
        recommendations.append(
            "😴 Maintain a healthier sleep schedule."
        )

    if sleep_hours > 9:
        recommendations.append(
            "⏰ Review your daily routine and maintain a consistent sleep schedule."
        )

    if not recommendations:
        recommendations.append(
            "🌟 Your input profile looks balanced. "
            "Continue maintaining consistent academic habits."
        )

    return recommendations


# ---------------- PREDICTION SECTION ----------------
if predict_button:

    input_data = pd.DataFrame([{
        "attendance": attendance,
        "previous_marks": previous_marks,
        "study_hours": study_hours,
        "assignment_completion": assignment_completion,
        "sleep_hours": sleep_hours
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))

    category = classify_performance(prediction)

    st.subheader("📊 Student Performance Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Final Marks",
        f"{prediction:.2f}/100"
    )

    col2.metric(
        "Performance Category",
        category
    )

    col3.metric(
        "Attendance",
        f"{attendance}%"
    )

    st.progress(int(prediction))

    if category == "Excellent":
        st.success("Excellent performance! Keep up the great work.")
    elif category == "Good":
        st.success("Good performance! You are progressing well.")
    elif category == "Average":
        st.info("Average performance. Consistent effort can improve your results.")
    else:
        st.warning("Needs improvement. Follow the recommendations below.")

    st.divider()

    # ---------------- PERSONALIZED RECOMMENDATIONS ----------------
    st.subheader("💡 Personalized Improvement Plan")

    recommendations = generate_recommendations(
        attendance,
        previous_marks,
        study_hours,
        assignment_completion,
        sleep_hours
    )

    for recommendation in recommendations:
        st.write("• " + recommendation)

    st.divider()

    # ---------------- STUDENT PROFILE ANALYTICS ----------------
    st.subheader("📈 Student Input Analysis")

    profile = pd.DataFrame({
        "Factor": [
            "Attendance",
            "Previous Marks",
            "Assignment Completion",
            "Study Hours (normalized)",
            "Sleep Hours (normalized)"
        ],
        "Score": [
            attendance,
            previous_marks,
            assignment_completion,
            study_hours / 12 * 100,
            sleep_hours / 12 * 100
        ]
    })

    fig_profile = px.bar(
        profile,
        x="Factor",
        y="Score",
        color="Score",
        range_y=[0, 100],
        title="Student Profile Overview (Normalized to 100)"
    )

    st.plotly_chart(fig_profile, use_container_width=True)

    st.caption(
        "Study and sleep hours are normalized to a 0–100 scale "
        "for visualization only."
    )

    st.divider()

# ---------------- MODEL COMPARISON ----------------
st.subheader("🤖 Machine Learning Model Comparison")

st.write(
    "The system evaluates three regression algorithms "
    "using standard model evaluation metrics."
)

st.dataframe(
    results.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "R2 Score": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

col1, col2 = st.columns(2)

with col1:
    fig_mae = px.bar(
        results,
        x="Model",
        y="MAE",
        color="Model",
        title="Mean Absolute Error (Lower is Better)"
    )
    st.plotly_chart(fig_mae, use_container_width=True)

with col2:
    fig_r2 = px.bar(
        results,
        x="Model",
        y="R2 Score",
        color="Model",
        title="R² Score (Higher is Better)"
    )
    st.plotly_chart(fig_r2, use_container_width=True)


# Feature Importance Analysis
st.header("📊 Feature Importance Analysis")

st.write(
    "This chart shows which factors have the "
    "greatest influence on student performance."
)

fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Factors Affecting Student Performance",
    color="Importance",
    color_continuous_scale="Blues"
)

fig_importance.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="Importance (Absolute Coefficient)",
    yaxis_title="Student Performance Factors"
)

st.plotly_chart(fig_importance, use_container_width=True)


# ---------------- FEATURE IMPORTANCE ----------------

st.divider()

st.header("📊 Feature Importance Analysis")

st.write(
    "Understand which factors contribute most "
    "to predicting student academic performance."
)

fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Blues",
    title="Impact of Different Student Factors"
)

fig_importance.update_layout(
    height=450,
    template="plotly_white",
    yaxis_title="Student Factors",
    xaxis_title="Feature Importance",
    showlegend=False
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# ---------------- ABOUT PROJECT ----------------
st.divider()

with st.expander("ℹ️ About This Project"):

    st.markdown("""
    ### Student Performance Prediction System

    **Objective:**

    To predict student academic performance using
    machine learning techniques.

    **Input Features:**
    - Attendance percentage
    - Previous academic marks
    - Daily study hours
    - Assignment completion
    - Daily sleep hours

    **Machine Learning Algorithms:**
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression

    **Evaluation Metrics:**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score

    **Technology Stack:**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly

    **Dataset:**

    This academic demonstration uses a synthetically
    generated dataset containing 500 student records.

    **Disclaimer:**

    Predictions are illustrative estimates based on
    synthetic training data and should not be treated
    as actual academic assessments.
    """)

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Python • Machine Learning • Streamlit"
)
