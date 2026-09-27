"""
Automated Test Suite for Customer Churn Prediction Flask API.
Uses standard library unittest.
Verifies endpoints, schema, mathematical properties of probabilities,
and prediction logic without hardcoding artificial outcomes.
"""

import sys
import os
import unittest
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


class FlaskApiTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        """GET /health must return 200 OK with 'healthy' status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("ready"))
        self.assertIn("timestamp", data)

    def test_home_page(self):
        """GET / must render the HTML dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ChurnGuard AI", response.data)
        self.assertIn(b"churnForm", response.data)

    def test_predict_schema_and_invariants(self):
        """
        Test POST /predict with high, low, and moderate risk profiles.
        Verifies API contract and mathematical correctness:
        - Probabilities in [0, 1]
        - churn_prob + non_churn_prob == 1.0
        - prediction is exactly 'Churn' or 'No Churn'
        - risk_level is in ['Low', 'Medium', 'High']
        - threshold consistency: churn_prob >= 0.5 <=> prediction == 'Churn'
        """
        test_cases = [
            {
                "name": "High Risk Customer",
                "payload": {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 2,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 95.50,
                    "TotalCharges": 191.00
                }
            },
            {
                "name": "Loyal / Low Risk Customer",
                "payload": {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "tenure": 65,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": 58.00,
                    "TotalCharges": 3770.00
                }
            },
            {
                "name": "Moderate Risk Customer",
                "payload": {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 24,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                    "Contract": "One year",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "MonthlyCharges": 76.50,
                    "TotalCharges": 1836.00
                }
            }
        ]

        for tc in test_cases:
            res = self.client.post("/predict", json=tc["payload"])
            self.assertEqual(res.status_code, 200, f"Failed on {tc['name']}: {res.data}")

            data = res.get_json()
            self.assertIsInstance(data, dict)

            # Check required fields
            self.assertIn("prediction", data)
            self.assertIn("churn_probability", data)
            self.assertIn("non_churn_probability", data)
            self.assertIn("risk_level", data)

            churn_p = data["churn_probability"]
            non_churn_p = data["non_churn_probability"]
            pred_label = data["prediction"]
            risk = data["risk_level"]

            # Mathematical and type assertions
            self.assertIsInstance(churn_p, (float, int))
            self.assertIsInstance(non_churn_p, (float, int))
            self.assertTrue(0.0 <= churn_p <= 1.0, f"churn_probability {churn_p} out of bounds")
            self.assertTrue(0.0 <= non_churn_p <= 1.0, f"non_churn_probability {non_churn_p} out of bounds")
            self.assertAlmostEqual(churn_p + non_churn_p, 1.0, places=3)

            # Valid categorical choices
            self.assertIn(pred_label, ["Churn", "No Churn"])
            self.assertIn(risk, ["Low", "Medium", "High"])

            # Decision boundary consistency
            if churn_p >= 0.5:
                self.assertEqual(pred_label, "Churn")
            else:
                self.assertEqual(pred_label, "No Churn")

            print(f"VERIFIED: {tc['name']} -> {pred_label} (Churn: {churn_p:.4f}, Non-Churn: {non_churn_p:.4f}, Risk: {risk})")


if __name__ == "__main__":
    unittest.main()
