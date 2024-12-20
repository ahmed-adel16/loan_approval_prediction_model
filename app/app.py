from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained Decision Tree Classifier model
model_path = os.path.join(os.path.dirname(__file__), 'dtc_model.joblib')
dtc = None

try:
    if os.path.exists(model_path):
        dtc = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if dtc is None:
        return render_template(
            'index.html', 
            prediction_text="Model is not available. Please check the server logs."
        )

    try:
        # Get user input from the form
        person_income = float(request.form.get('person_income', 0))
        previous_loan_defaults_on_file = int(request.form.get('previous_loan_defaults_on_file', 0))
        loan_percent_income = float(request.form.get('loan_percent_income', 0))
        loan_int_rate = float(request.form.get('loan_int_rate', 0))

        # Create input array for the model
        input_features = np.array([[person_income, previous_loan_defaults_on_file, loan_percent_income, loan_int_rate]])

        # Predict the outcome using the model
        prediction = dtc.predict(input_features)

        # Map prediction to readable output and corresponding class
        if prediction[0] == 1:
            outcome = 'Approved'
            outcome_class = 'approved'
        else:
            outcome = 'Denied'
            outcome_class = 'denied'

        return render_template(
            'index.html', 
            prediction_text=outcome,
            prediction_class=outcome_class
        )
    except ValueError as ve:
        return render_template(
            'index.html', 
            prediction_text=f"Input Error: {str(ve)}",
            prediction_class="error"
        )
    except Exception as e:
        return render_template(
            'index.html', 
            prediction_text=f"Unexpected Error: {str(e)}",
            prediction_class="error"
        )

if __name__ == '__main__':
    app.run(debug=True)
