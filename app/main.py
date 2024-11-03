import os
import sys
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
# Add the parent directory to the system path
sys.path.append(os.path.abspath(".."))
from config import OPENAICONFIGPARAMETERS as of

engine = 'gpt-4o-mini'
client = OpenAI(api_key=of.api_key, organization=of.organization_id, project=of.project_id)

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

def get_llm_summary(client, df):
    top_contributors = df.head(5)
    
    # Generate the prompt
    prompt = f"""
    Please explain to the client the top 5 features contributing to the model's prediction, based on SHAP values. The SHAP values represent each feature's impact on the prediction, with higher absolute values indicating a stronger influence. Here is a breakdown of the top 5 contributors:
    
    1. **{top_contributors.iloc[0]['Feature']}**: 
       - SHAP Value: {top_contributors.iloc[0]['SHAP Value (Log-Odds)']}
       - Absolute Impact: {top_contributors.iloc[0]['SHAP Value Absolute Value']}
       - Feature Value: {top_contributors.iloc[0]['Values']}
    
    2. **{top_contributors.iloc[1]['Feature']}**: 
       - SHAP Value: {top_contributors.iloc[1]['SHAP Value (Log-Odds)']}
       - Absolute Impact: {top_contributors.iloc[1]['SHAP Value Absolute Value']}
       - Feature Value: {top_contributors.iloc[1]['Values']}
    
    3. **{top_contributors.iloc[2]['Feature']}**: 
       - SHAP Value: {top_contributors.iloc[2]['SHAP Value (Log-Odds)']}
       - Absolute Impact: {top_contributors.iloc[2]['SHAP Value Absolute Value']}
       - Feature Value: {top_contributors.iloc[2]['Values']}
    
    4. **{top_contributors.iloc[3]['Feature']}**: 
       - SHAP Value: {top_contributors.iloc[3]['SHAP Value (Log-Odds)']}
       - Absolute Impact: {top_contributors.iloc[3]['SHAP Value Absolute Value']}
       - Feature Value: {top_contributors.iloc[3]['Values']}
    
    5. **{top_contributors.iloc[4]['Feature']}**: 
       - SHAP Value: {top_contributors.iloc[4]['SHAP Value (Log-Odds)']}
       - Absolute Impact: {top_contributors.iloc[4]['SHAP Value Absolute Value']}
       - Feature Value: {top_contributors.iloc[4]['Values']}
    
    These features had the most significant influence on the prediction outcome, with positive or negative SHAP values indicating the direction of their effect.
    """
    completions = client.chat.completions.create(model=engine, 
                                                 temperature=0, 
                                                 messages=[{"role": "user", "content": prompt}])
    summary = completions.choices[0].message.content
    return summary

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

    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer(transformed_df)

    explanations = []
    for idx, row in transformed_df.iterrows():
        test_point = transformed_df.iloc[[idx], :].transpose().reset_index()
        test_point.columns = ['Feature', 'Values']

        # SHAP values for each feature
        shap_values_instance = shap_values.values[idx]
        feature_names = transformed_df.columns
        # Create a DataFrame to display each feature's SHAP value and probability impact
        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value (Log-Odds)': shap_values_instance,
            'SHAP Value Absolute Value': np.abs(shap_values_instance)
        })
        df_shap = df_shap.sort_values(by='SHAP Value Absolute Value', ascending=False).reset_index(drop=True)
        df_shap = df_shap.merge(test_point, on='Feature')
        explanation = get_llm_summary(client, df_shap)
        explanations.append(explanation)
    
    
    return {"predictions": y_pred.tolist(), "explanations": explanations}
