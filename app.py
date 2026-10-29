import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("knn_heart.pkl")
SCALER_PATH = Path("scaler.pkl")
COLUMNS_PATH = Path("columns.pkl")

APP_TITLE = "Heart Disease Prediction"
APP_SUBTITLE = "by Akarsh"

SEX_OPTIONS = ["Male", "Female"]
CHEST_PAIN_OPTIONS = ["ATA", "NAP", "TA", "ASY"]
RESTING_ECG_OPTIONS = ["Normal", "ST", "LVH"]
EXERCISE_ANGINA_OPTIONS = ["Yes", "No"]
ST_SLOPE_OPTIONS = ["Up", "Flat", "Down"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts() -> tuple[Any, Any, list[str]]:
    missing = [p for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifact(s): {', '.join(str(p) for p in missing)}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, expected_columns


def build_input_frame(
    age: int,
    sex: str,
    chest_pain: str,
    resting_bp: int,
    cholesterol: int,
    fasting_bs: int,
    resting_ecg: str,
    max_hr: int,
    exercise_angina: str,
    oldpeak: float,
    st_slope: str,
    expected_columns: list[str],
) -> pd.DataFrame:
    raw_input: dict[str, Any] = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[expected_columns]


def render_form() -> dict[str, Any]:
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", SEX_OPTIONS)
        chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN_OPTIONS)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

    with col2:
        resting_ecg = st.selectbox("Resting ECG", RESTING_ECG_OPTIONS)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA_OPTIONS)
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ST_SLOPE_OPTIONS)

    return {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
    }


def render_prediction(model: Any, scaler: Any, input_df: pd.DataFrame) -> None:
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    if probability is not None:
        st.metric("Estimated Risk Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "This tool provides a statistical estimate only and is not a medical "
        "diagnosis. Please consult a qualified healthcare professional."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="❤️", layout="centered")
    st.title(f"❤️ {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.markdown("Provide the following details to check your heart disease risk:")

    try:
        model, scaler, expected_columns = load_artifacts()
    except FileNotFoundError as e:
        logger.error(e)
        st.error(f"Could not load model files: {e}")
        st.stop()

    inputs = render_form()

    if st.button("Predict", type="primary"):
        try:
            input_df = build_input_frame(**inputs, expected_columns=expected_columns)
            render_prediction(model, scaler, input_df)
        except Exception as e:
            logger.exception("Prediction failed")
            st.error(f"Something went wrong while predicting: {e}")


if __name__ == "__main__":
    main()