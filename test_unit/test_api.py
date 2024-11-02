import requests
import json

url = "http://127.0.0.1:8000/predict"


features = [
    # Test 1 example with Fraud = 1
    {
        "step": 62,
        "type": "TRANSFER",
        "amount": 401529.35,
        "nameOrig": "C330207825",
        "oldbalanceOrg": 401529.35,
        "newbalanceOrig": 0.0,
        "nameDest": "C1825454306",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    },
    # Test 2 example with Fraud = 0
    {
        "step": 278,
        "type": "CASH_IN",
        "amount": 330218.42,
        "nameOrig": "C632336343",
        "oldbalanceOrg": 20866.0,
        "newbalanceOrig": 351084.42,
        "nameDest": "C834976624",
        "oldbalanceDest": 452419.57,
        "newbalanceDest": 122201.15
    }
]

# Send a POST request with JSON data
response = requests.post(url, json=features)

# Parse the response JSON
answer = response.json()

print(answer)
