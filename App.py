"""Simple student performance prediction using a RandomForestRegressor.

This script can be executed in two modes:

1. **Command-line** (original behavior) - the user is prompted for values.
2. **Streamlit UI** - run with ``streamlit run Main.py`` and a web form is shown.

The dataset must be available at the path used below or you can adjust the
location to where your ``StudentPerformanceFactors.csv`` lives.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# we will import streamlit lazily since the CLI mode doesn't need it
try:
    import streamlit as st
except ImportError:
    st = None


# ----------- training logic -----------
def train_model(csv_path: str):
    # load and prepare data
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()
    X = data[
        [
            "Hours_Studied",
            "Attendance",
            "Sleep_Hours",
            "Previous_Scores",
            "Tutoring_Sessions",
            "Physical_Activity",
        ]
    ]
    y = data["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def predict(model, features):
    # features is a list or array-like with six numeric values
    return model.predict([features])[0]


def run_cli(model):
    print("Student Performance Prediction Model (CLI)")
    A = float(input("Hours_Studied: "))
    B = float(input("Attendance: "))
    C = float(input("Sleep_Hours: "))
    D = float(input("Previous_Scores: "))
    E = float(input("Tutoring_Sessions: "))
    F = float(input("Physical_Activity: "))
    pred = predict(model, [A, B, C, D, E, F])
    print("Predicted Exam Score: ", pred)


def run_streamlit(model):
    # the streamlit app will only be available when "st" is not None
    # page configuration for better visuals
    st.set_page_config(
        page_title="Student Performance Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # custom CSS for vibrant UI with background images
    st.markdown(
        """
        <style>
        /* Full-page background with study-themed images */
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
                        url('https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?auto=format&fit=crop&w=1920&q=80'),
                        url('https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&w=1920&q=80');
            background-size: cover, cover;
            background-position: center, center;
            background-attachment: fixed;
            color: white;
        }

        /* Style sidebar */
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.9);
            color: black;
        }

        /* Style selectboxes */
        .stSelectbox > div > div > div > div {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
            border-radius: 8px;
            padding: 8px;
        }

        /* Style button */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }

        /* Style success message */
        .stSuccess {
            background-color: rgba(76, 175, 80, 0.8);
            color: white;
            border-radius: 8px;
            padding: 10px;
        }

        /* Title styling */
        h1 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        /* Big prediction display */
        .big-prediction {
            font-size: 48px;
            font-weight: bold;
            color: yellow;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and intro
    st.title("🎓 Student Performance Predictor")
    st.markdown("### Predict your exam score based on study habits!")
    st.markdown("Use the sidebar to input your details, then click **Predict**.")

    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Input Your Details")
        
        hours = st.selectbox(
            "⏰ Hours Studied per day",
            options=[f"{i * 0.5:.1f}" for i in range(0, 25)],
            index=0,
            help="How many hours do you study daily?"
        )
        attendance = st.selectbox(
            "📊 Attendance (%)",
            options=[str(i) for i in range(0, 101)],
            index=80,  # default to 80%
            help="Your class attendance percentage."
        )
        sleep = st.selectbox(
            "😴 Sleep Hours per night",
            options=[f"{i * 0.5:.1f}" for i in range(0, 49)],
            index=16,  # default to 8 hours
            help="Average sleep hours."
        )
        prev = st.selectbox(
            "📈 Previous Scores",
            options=[str(i) for i in range(0, 101)],
            index=70,  # default to 70%
            help="Your previous exam scores."
        )
        tutoring = st.selectbox(
            "👨‍🏫 Tutoring Sessions per week",
            options=[str(i) for i in range(0, 21)],
            index=0,
            help="Number of tutoring sessions."
        )
        activity = st.selectbox(
            "⚽ Physical Activity (hours/week)",
            options=[str(i) for i in range(0, 21)],
            index=2,  # default to 2 hours
            help="Hours of physical activity."
        )

        predict_button = st.button("🔮 Predict Score", use_container_width=True)

    # Main area for results
    if predict_button:
        with st.spinner("Calculating your predicted score..."):
            import time
            time.sleep(1)  # simulate processing
            score = predict(model, [float(hours), float(attendance), float(sleep), float(prev), float(tutoring), float(activity)])
        
        # Highlight the prediction big
        st.markdown(f'<div class="big-prediction">🎉 Your predicted exam score is: {score:.2f}</div>', unsafe_allow_html=True)
        
        # Add some visual feedback
        if score >= 80:
            st.balloons()
            st.markdown("🌟 Excellent! Keep up the great work!")
        elif score >= 60:
            st.markdown("👍 Good job! There's room for improvement.")
        else:
            st.markdown("💪 Don't worry, focus on study habits and try again!")

        # Optional: Display a chart or something
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            labels = ['Hours Studied', 'Attendance', 'Sleep', 'Previous Scores', 'Tutoring', 'Activity']
            values = [float(hours), float(attendance), float(sleep), float(prev), float(tutoring), float(activity)]
            ax.bar(labels, values, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
            ax.set_ylabel('Values')
            ax.set_title('Your Input Factors')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except ImportError:
            st.warning("Matplotlib not available for chart display.")


if __name__ == "__main__":
    # modify the path below if your CSV is stored elsewhere
    # For cloud deployment, place the CSV in the same directory as Main.py
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "StudentPerformanceFactors.csv")
    if not os.path.exists(csv_path):
        # Fallback to the original path for local Windows
        csv_path = "C:\\Users\\Raviteja Ramisetty\\Downloads\\StudentPerformanceFactors.csv"
    model = train_model(csv_path)
    if st:
        run_streamlit(model)
    else:
        run_cli(model)
