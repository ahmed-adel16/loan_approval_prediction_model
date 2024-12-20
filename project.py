# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('loan_data.csv')
df = pd.DataFrame(data)

# Check for null values
df.isnull().sum()

# Describe the dataset
df.describe(include='all')

# Encoding categorical columns
def encoding(col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Removing outliers
def remove_outlier(col):
    global df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    df = df.loc[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

# Identify columns to process
cols = ['person_gender', 'person_home_ownership', 'person_education', 'loan_intent',
        'previous_loan_defaults_on_file']
outliers = ['person_age', 'person_income',
            'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'credit_score']

# Remove outliers and encode categorical variables
for col in outliers:
    remove_outlier(col)
for col in cols:
    encoding(col)

# Drop duplicates and NaN values
df.dropna()
df.drop_duplicates()

# Split features and target variable
x = df.drop(columns=['loan_status'])
y = df['loan_status']

# Recursive Feature Elimination (RFE) for feature selection
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=4)
rfe.fit(x, y)

# Print feature selection results
for i, col in zip(range(x.shape[1]), x.columns):
    print(f'{col} selected: {rfe.support_[i]}, rank = {rfe.ranking_[i]}')

# Summarize feature rankings in a DataFrame
features = x.columns
ranks = rfe.ranking_
selected = rfe.support_

rfe_results = pd.DataFrame({
    'feature': features,
    'ranks': ranks,
    'selected': selected
})
rfe_results.sort_values(by='ranks', inplace=True)

# Select the top features
df = df[['person_income', 'previous_loan_defaults_on_file', 'loan_percent_income', 'loan_int_rate']]

# Split data into train, test, and validation sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=12)

# Define a function to evaluate a model
def apply_model(model):
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)
    print('Train Classification Report: ')
    print(classification_report(y_train, train_pred))
    print('Test Classification Report: ')
    print(classification_report(y_test, test_pred))

# Apply Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=12)
apply_model(dtc)

lr_model = LogisticRegression()
apply_model(lr_model)