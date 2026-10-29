
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Use the filenames present in your project folder.
MODEL_PATH = BASE_DIR / "KNN_heart.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

# Load the existing model and preprocessing artifacts.
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_columns = list(joblib.load(COLUMNS_PATH))


def build_input_frame(form):
    raw_input = {
        "Age": int(form["age"]),
        "RestingBP": int(form["trtbps"]),
        "Cholesterol": int(form["chol"]),
        "FastingBS": int(form["fbs"]),
        "MaxHR": int(form["thalach"]),
        "Oldpeak": float(form["oldpeak"]),
        f"Sex_{form['sex']}": 1,
        f"ChestPainType_{form['cp']}": 1,
        f"RestingECG_{form['restecg']}": 1,
        f"ExerciseAngina_{form['exang']}": 1,
        f"ST_Slope_{form['slope']}": 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Match the original model's expected columns and their order.
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    return input_df[expected_columns]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            input_df = build_input_frame(request.form)
            scaled_input = scaler.transform(input_df)
            prediction = int(model.predict(scaled_input)[0])

            if prediction == 1:
                result = "High Risk of Heart Disease"
            else:
                result = "Low Risk of Heart Disease"

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_input)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = float(
                        probabilities[classes.index(1)]
                    )

        except (KeyError, ValueError, TypeError) as exc:
            error = f"Please check the entered values: {exc}"
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check the model files "
                "and the expected input columns."
            )

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
