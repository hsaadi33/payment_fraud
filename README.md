# Payment Fraud Detection Project

This project aims to detect fraudulent payment transactions using a machine learning model. The project involves data exploration, model training with the CatBoost algorithm, and an API built using FastAPI to serve predictions, and model explanability by using shap values and wrapping them with an LLM explanation. The API can be tested locally using the provided `test_api.py` script. You can get the dataset from kaggle (https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset).

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Exploration and Modeling](#data-exploration-and-modeling)
- [API Usage and Testing](#api-usage-testing)
- [Future Improvements](#future-improvements)

## Project Structure
```plaintext
payment_fraud/
├── app/
│   ├── main.py               # FastAPI app for fraud prediction
│   ├── __init__              
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis notebook
│   ├── modeling.ipynb        # Model training notebook
├── test_unit/
│   ├── test_api.py           # Script to test the API locally
├── requirements.txt          # Required Python packages
├── DockerFile                # DockerFile
└── README.md                 # Project documentation
```

## Data Exploration and Modeling
The notebooks/eda.ipynb file contains exploratory data analysis (EDA) on transaction data, identifying key patterns and correlations useful for fraud detection. We did feature engineering based on this EDA.

The modeling.ipynb file covers data preprocessing, and model training using the CatBoost algorithm. The trained model is saved for use in the API. Also, we include shap values that are wrapped by an LLM explanation so that the client understands why this prediction was made by the most important features.

## API Usage and Testing
The FastAPI app serves the trained model for predictions. To start the API locally, change the directory to the app directory, run the fastapi app in main.py. To test the api, open another terminal window, and change the directory to test_api, and then run python test_unit/test_api.py.


## Future Improvements
- Enhance the model with additional features and tuning.
- Deploy the API to a cloud service for production use.