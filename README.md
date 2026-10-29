# PulsePredict — Heart Disease Prediction Web Application

PulsePredict is an educational machine learning web application that predicts a heart disease risk category based on user-provided health attributes.

The application uses a trained K-Nearest Neighbors (KNN) classification model, Python, Flask, Pandas, and Scikit-learn.

## Features

- Interactive web form for entering health attributes
- Machine learning predictions using a trained KNN classifier
- Feature scaling and categorical input handling
- Model-generated prediction results
- Web interface built with HTML and CSS
- Flask backend for processing inputs and displaying results

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- Joblib
- HTML and CSS
- K-Nearest Neighbors (KNN)

## Project Structure

```text
PulsePredict/
├── flask_app.py
├── app.py
├── KNN_heart.pkl
├── scaler.pkl
├── columns.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── .gitignore
└── README.md
```

## How It Works

1. The user enters health-related attributes through the web form.
2. Flask receives and processes the submitted values.
3. The application prepares the input features in the expected column order.
4. The saved scaler transforms the input data.
5. The trained KNN model generates a prediction.
6. The application displays the result on the webpage.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/pulse-predict.git
cd pulse-predict
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
python -m pip install flask pandas scikit-learn joblib
```

Use compatible library versions for the saved model artifacts.

### 5. Start the Flask application

```bash
python flask_app.py
```

Open the local URL shown in the terminal, typically `http://127.0.0.1:5000`.

## Important Disclaimer

PulsePredict is an educational project, not a clinically validated medical tool. Its predictions may be inaccurate and must not be used for diagnosis or healthcare decisions. Consult a qualified healthcare professional for medical advice.

## Acknowledgment

This project was developed by adapting and extending an existing heart disease prediction project:

Original repository: https://github.com/Abhishekmathanagi/heart-attack-detection_ML

Please refer to the original repository's license and retain any attribution required by its terms. The Flask integration and other modifications should be described as your own contributions only where applicable.
