import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize logging with file handler
class Logger:
    log_file_path = "app_logs.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path)]
    )
    logger = logging.getLogger(__name__)

# Add the parent directory to the system path
sys.path.append(os.path.abspath(".."))
from config import OPENAICONFIGPARAMETERS as of

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=of.api_key, organization=of.organization_id, project=of.project_id)
        self.engine = 'gpt-4o-mini'

    def generate_summary(self, df):
        Logger.logger.info("Generating LLM summary for SHAP values.")
        try:
            top_contributors = df.head(5)
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
            completions = self.client.chat.completions.create(
                model=self.engine, temperature=0, messages=[{"role": "user", "content": prompt}]
            )
            summary = completions.choices[0].message.content
            Logger.logger.info("LLM summary generated successfully.")
            return summary
        except Exception as e:
            Logger.logger.error("LLM summary generation failed: %s", e, exc_info=True)
            raise


class FraudModel:
    model_path = os.path.join(os.path.dirname(__file__), "../notebooks/catboost_model.pkl")

    def __init__(self):
        self.model = self.load_model()
        self.explainer = shap.TreeExplainer(self.model)

    def load_model(self):
        try:
            Logger.logger.info("Loading CatBoost model from path: %s", self.model_path)
            with open(self.model_path, "rb") as file:
                model = pickle.load(file)
            Logger.logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            Logger.logger.error("Failed to load model: %s", e, exc_info=True)
            raise

    def transform_dataframe(self, df):
        Logger.logger.debug("Starting DataFrame transformation.")
        try:
            df = df.copy()
            df["hour"] = (df["step"] - 1) % 24
            df["orig_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
            df["dest_balance_change"] = df["oldbalanceDest"] - df["newbalanceDest"]
            df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
            df["is_zero_balance_after"] = (df["newbalanceOrig"] == 0).astype(int)
            high_value_threshold = df["amount"].quantile(0.95)
            df["high_value_transfer"] = (df["amount"] > high_value_threshold).astype(int)
            df["dest_account_type"] = df["nameDest"].str[0]
            df["same_balance_before"] = (df["oldbalanceOrg"] == df["oldbalanceDest"]).astype(int)
            df["same_balance_after"] = (df["newbalanceOrig"] == df["newbalanceDest"]).astype(int)
        except Exception as e:
            Logger.logger.error("DataFrame transformation failed: %s", e, exc_info=True)
            raise

        features = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'hour',
            'orig_balance_change', 'dest_balance_change', 'amount_to_orig_balance_ratio', 'type',
            'is_zero_balance_after', 'high_value_transfer', 'dest_account_type', 'same_balance_before',
            'same_balance_after'
        ]
        return df[features]

    def predict(self, df):
        transformed_df = self.transform_dataframe(df)
        y_pred_proba = self.model.predict_proba(transformed_df)
        y_pred = np.argmax(y_pred_proba, axis=1)
        shap_values = self.explainer(transformed_df)
        return y_pred, shap_values, transformed_df

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

class FraudDetectionAPI:
    def __init__(self):
        self.app = FastAPI()
        Instrumentator().instrument(self.app).expose(self.app, endpoint="/metrics")
        self.model = FraudModel()
        self.openai_client = OpenAIClient()
        self.setup_routes()

    def setup_routes(self):
        self.app.get("/")(self.health_check)
        self.app.post("/predict")(self.predict)

    @staticmethod
    def health_check():
        return {"health_check": "OK"}

    async def predict(self, transactions: List[Transaction]):
        Logger.logger.info(f"Received prediction request with {len(transactions)} transactions.")
        try:
            df = pd.DataFrame([transaction.dict() for transaction in transactions])
            y_pred, shap_values, transformed_df = self.model.predict(df)
            feature_names = transformed_df.columns[:len(shap_values.values[0])]
            explanations = []

            for idx, row in transformed_df.iterrows():
                test_point = transformed_df.iloc[[idx], :].transpose().reset_index()
                test_point.columns = ['Feature', 'Values']
            
                df_shap = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value (Log-Odds)': shap_values.values[idx][:len(feature_names)],
                        'SHAP Value Absolute Value': np.abs(shap_values.values[idx][:len(feature_names)])
                    })
                df_shap = df_shap.sort_values(by='SHAP Value Absolute Value', ascending=False).reset_index(drop=True)
                df_shap = df_shap.merge(test_point, on='Feature', how='left')
                explanations.append(self.openai_client.generate_summary(df_shap))


            Logger.logger.info("Predictions and explanations generated successfully.")
            return {"predictions": y_pred.tolist(), "explanations": explanations}
        except Exception as e:
            Logger.logger.error("Prediction failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred during prediction.")

app_instance = FraudDetectionAPI()
app = app_instance.app
