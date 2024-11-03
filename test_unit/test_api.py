import requests
import json

def test_fraud_detection_api():
    url = "http://127.0.0.1:8000/predict"
    
    # Define test cases
    test_cases = [
        {
            "input": {
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
            "expected_prediction": 1
        },
        {
            "input": {
                "step": 278,
                "type": "CASH_IN",
                "amount": 330218.42,
                "nameOrig": "C632336343",
                "oldbalanceOrg": 20866.0,
                "newbalanceOrig": 351084.42,
                "nameDest": "C834976624",
                "oldbalanceDest": 452419.57,
                "newbalanceDest": 122201.15
            },
            "expected_prediction": 0
        }
    ]

    # Prepare the input data for the API request
    features = [case["input"] for case in test_cases]
    
    try:
        # Send POST request
        response = requests.post(url, json=features)
        response.raise_for_status()  # Raise an error for unsuccessful requests
        
        # Parse JSON response
        response_data = response.json()
        
        # Extract predictions and explanations
        predictions = response_data.get("predictions", [])
        explanations = response_data.get("explanations", [])
        
        # Validate predictions and explanations length
        assert len(predictions) == len(test_cases), f"Expected {len(test_cases)} predictions, got {len(predictions)}"
        assert len(explanations) == len(test_cases), f"Expected {len(test_cases)} explanations, got {len(explanations)}"

        # Check each prediction against the expected value
        for i, case in enumerate(test_cases):
            assert predictions[i] == case["expected_prediction"], f"Test case {i + 1} failed: expected {case['expected_prediction']}, got {predictions[i]}"
        
        # If all assertions pass, print predictions and explanations
        print("All tests passed successfully!")
        print("Predictions:", predictions)
        print("Explanations:", explanations)
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
    except AssertionError as e:
        print(f"Assertion error: {e}")

# Run the test function
if __name__ == "__main__":
    test_fraud_detection_api()
