import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'loan_data.csv')

try:
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}.")
    exit()

# Encode categorical columns
def encoding(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Remove outliers
def remove_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    return df.loc[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

# Process columns
cols = ['person_gender', 'person_home_ownership', 'person_education', 'loan_intent', 'previous_loan_defaults_on_file']
outliers = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'credit_score']

# Remove outliers from DataFrame
for col in outliers:
    data = remove_outlier(data, col)

# Encode categorical columns
for col in cols:
    encoding(data, col)

# Split features and target variable
x = data[['person_income', 'previous_loan_defaults_on_file', 'loan_percent_income', 'loan_int_rate']]
y = data['loan_status']

# Train the Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=12)
dtc.fit(x, y)

# Save the model as 'dtc_model.joblib' using joblib
model_path = os.path.join(os.path.dirname(__file__), 'dtc_model.joblib')
joblib.dump(dtc, model_path)

print(f"Model has been trained and saved as '{model_path}'.")
