"""
Data Preprocessing Module for Customer Churn Prediction.
Faithfully reproduces the Codebasics Telco Churn preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling',
    'MonthlyCharges',
    'TotalCharges',
    'InternetService_DSL',
    'InternetService_Fiber optic',
    'InternetService_No',
    'Contract_Month-to-month',
    'Contract_One year',
    'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

COLS_TO_SCALE = ['tenure', 'MonthlyCharges', 'TotalCharges']

YES_NO_COLUMNS = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Drops customerID
    - Coerces TotalCharges to numeric, removes blank spaces
    - Standardizes 'No internet service' and 'No phone service' to 'No'
    - Maps binary categories to 0 and 1
    - One-hot encodes multi-class categoricals
    """
    df = df.copy()

    if 'customerID' in df.columns:
        df.drop('customerID', axis='columns', inplace=True)

    # TotalCharges cleaning
    if 'TotalCharges' in df.columns:
        df = df[df['TotalCharges'].astype(str).str.strip() != '']
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Standardize 'No internet service' and 'No phone service'
    df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

    # Yes/No columns
    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Churn target if present
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Gender: Female -> 1, Male -> 0
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0}).fillna(0).astype(int)

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen']).astype(int)

    # One-hot encoding for multi-class columns
    categorical_cols = [c for c in ['InternetService', 'Contract', 'PaymentMethod'] if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(data=df, columns=categorical_cols, dtype=int)

    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Full training data preparation pipeline:
    Cleans data, separates features & target, fits MinMaxScaler,
    and returns scaled X, y, scaler, and feature list.
    """
    df_clean = clean_dataframe(df)

    if 'Churn' not in df_clean.columns:
        raise ValueError('Dataset must contain Churn target column.')

    X = df_clean.drop('Churn', axis='columns')
    y = df_clean['Churn']

    # Ensure all required feature columns exist and are ordered identically
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS].copy()

    # Fit MinMaxScaler on tenure, MonthlyCharges, TotalCharges
    scaler = MinMaxScaler()
    X[COLS_TO_SCALE] = scaler.fit_transform(X[COLS_TO_SCALE])

    return X, y, scaler, FEATURE_COLUMNS


def transform_single_customer(data: dict, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transforms raw dictionary input from web form / JSON API into a scaled
    26-element numpy array matching model input shape (1, 26).
    """
    gender = 1 if str(data.get('gender', '')).lower() in ['female', '1'] else 0
    senior = 1 if str(data.get('SeniorCitizen', '')).lower() in ['1', 'yes', 'true'] else 0
    partner = 1 if str(data.get('Partner', '')).lower() in ['1', 'yes', 'true'] else 0
    dependents = 1 if str(data.get('Dependents', '')).lower() in ['1', 'yes', 'true'] else 0

    try:
        tenure = float(data.get('tenure', 0))
    except (ValueError, TypeError):
        tenure = 0.0

    phone_service = 1 if str(data.get('PhoneService', '')).lower() in ['1', 'yes', 'true'] else 0
    multiple_lines = 1 if str(data.get('MultipleLines', '')).lower() in ['1', 'yes', 'true'] else 0

    internet_service = str(data.get('InternetService', 'No'))
    online_security = 1 if str(data.get('OnlineSecurity', '')).lower() in ['1', 'yes', 'true'] else 0
    online_backup = 1 if str(data.get('OnlineBackup', '')).lower() in ['1', 'yes', 'true'] else 0
    device_protection = 1 if str(data.get('DeviceProtection', '')).lower() in ['1', 'yes', 'true'] else 0
    tech_support = 1 if str(data.get('TechSupport', '')).lower() in ['1', 'yes', 'true'] else 0
    streaming_tv = 1 if str(data.get('StreamingTV', '')).lower() in ['1', 'yes', 'true'] else 0
    streaming_movies = 1 if str(data.get('StreamingMovies', '')).lower() in ['1', 'yes', 'true'] else 0

    contract = str(data.get('Contract', 'Month-to-month'))
    paperless_billing = 1 if str(data.get('PaperlessBilling', '')).lower() in ['1', 'yes', 'true'] else 0
    payment_method = str(data.get('PaymentMethod', 'Electronic check'))

    try:
        monthly_charges = float(data.get('MonthlyCharges', 0.0))
    except (ValueError, TypeError):
        monthly_charges = 0.0

    raw_total = data.get('TotalCharges', None)
    if raw_total is not None and str(raw_total).strip() != '':
        try:
            total_charges = float(raw_total)
        except (ValueError, TypeError):
            total_charges = round(tenure * monthly_charges, 2)
    else:
        total_charges = round(tenure * monthly_charges, 2)

    # Base feature dictionary
    record = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service in ['No', 'None', ''] else 0,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    }

    df_single = pd.DataFrame([record])[FEATURE_COLUMNS]

    # Scale the 3 numeric features
    df_single[COLS_TO_SCALE] = scaler.transform(df_single[COLS_TO_SCALE])

    return df_single.to_numpy(dtype=np.float32)
