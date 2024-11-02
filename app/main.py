import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Define model path
model_path = os.path.join(os.path.dirname(__file__), "../notebooks/catboost_model.pkl")

# Load the CatBoost model
with open(model_path, "rb") as file:
    catboost_model = pickle.load(file)

# Create FastAPI app
app = FastAPI()

# Define the JSON schema as a Pydantic model
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

# Function to transform DataFrame to target format
def transform_dataframe(df):
    # Feature engineering
    # Create an `hour` column instead of `step` column because it might be useful to see how fraud payment occur at an hourly level in a day.
    df["hour"] = (df["step"] - 1) % 24

    # Balance Change: Calculate the change in the origin and destination balances
    df["orig_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_change"] = df["oldbalanceDest"] - df["newbalanceDest"]

    # Balance-to-Transaction Ratio: The ratio of the transaction amount to the origin account balance
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Is Zero Balance After Transaction: Flag cases where the origin account's balance is zero immediately after the transaction
    df["is_zero_balance_after"] = (df["newbalanceOrig"] == 0).astype(int)

    # High-Value Transfer: Flag transactions that are above a certain threshold (e.g., 95th percentile) as potentially suspicious
    high_value_threshold = df["amount"].quantile(0.95)
    df["high_value_transfer"] = (df["amount"] > high_value_threshold).astype(int)

    # dest_account_type: the destination account starts with ‘M’, as merchants are less likely to engage in fraud
    df["dest_account_type"] = df["nameDest"].str[0]

    # Orig/Dest Same Balance: Check if the origin and destination balances are the same before or after the transaction, which may indicate balance masking
    df["same_balance_before"] = (df["oldbalanceOrg"] == df["oldbalanceDest"]).astype(int)
    df["same_balance_after"] = (df["newbalanceOrig"] == df["newbalanceDest"]).astype(int)

    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'hour', 'orig_balance_change', 'dest_balance_change', 'amount_to_orig_balance_ratio']
    categorical_features = ['type', 'is_zero_balance_after', 'high_value_transfer', 'dest_account_type', 'same_balance_before', 'same_balance_after']
    
    features = numerical_features + categorical_features
    df_features = df[features].copy()
    
    return df_features

# Health check endpoint
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

# Prediction endpoint that accepts validated JSON data
@app.post("/predict")
async def predict(transactions: List[Transaction]):
    # Convert validated data to DataFrame
    df = pd.DataFrame([transaction.dict() for transaction in transactions])
    
    # Transform the DataFrame to include engineered features
    transformed_df = transform_dataframe(df)
    
    # Predict probabilities
    y_pred_proba = catboost_model.predict_proba(transformed_df)
    
    # Convert probabilities to predicted class
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return {"predictions": y_pred.tolist()}
