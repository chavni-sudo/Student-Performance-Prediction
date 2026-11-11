# Student Performance Prediction System

## 1. Project Overview

The Student Performance Prediction System is an AI/ML-based application developed using Python. It predicts student academic performance based on various academic and lifestyle factors.

The system uses machine learning algorithms to analyze student-related information and generate performance predictions through an interactive web dashboard.

## 2. Problem Statement

Academic performance depends on several factors, including attendance, previous academic results, study hours, assignment completion and sleeping habits.

This project aims to use machine learning techniques to identify patterns in these factors and predict student performance, helping identify students who may require additional academic support.

## 3. Objectives

- Predict student academic performance using machine learning.
- Analyze the impact of academic and lifestyle factors.
- Compare different machine learning algorithms.
- Develop an interactive and user-friendly dashboard.
- Demonstrate the practical application of AI and ML in education.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 5. Machine Learning Algorithms

The project implements and compares three machine learning algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor

## 6. Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Comparison Results

| Algorithm | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.6617 | 4.7050 | 0.8259 |
| Random Forest | 4.5606 | 5.7505 | 0.7399 |
| Decision Tree | 6.6611 | 8.4994 | 0.4317 |

Linear Regression achieved the best performance among the three evaluated models on the current test data.

## 7. Input Features

The application accepts the following student details:

- Attendance Percentage
- Previous Marks
- Daily Study Hours
- Assignment Completion Percentage
- Daily Sleep Hours

## 8. Project Structure

```text
Student_Performance_Prediction/
│
├── data/
│   └── student_performance.csv
│
├── models/
│   └── trained model files
│
├── venv/
│
├── app.py
├── train_model.py
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project directory

```bash
cd Student_Performance_Prediction
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

## 10. Application Features

- Interactive student input panel.
- Machine learning-based performance prediction.
- Comparison of multiple ML algorithms.
- Model evaluation metrics.
- Graphical representation of model performance.

## 11. Future Scope

- Integration of larger student datasets.
- More advanced machine learning algorithms.
- Early identification of academically at-risk students.
- Personalized academic improvement recommendations.
- Cloud-based deployment.
- Integration with educational management systems.

## 12. Conclusion

The Student Performance Prediction System demonstrates how artificial intelligence and machine learning can be applied to educational data.

By analyzing academic and lifestyle factors, the application provides an accessible way to estimate student performance and explore the relationship between different factors and academic outcomes.

---

**Project Domain:** Artificial Intelligence and Machine Learning

**Programming Language:** Python

**Application Framework:** Streamlit