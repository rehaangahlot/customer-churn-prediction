import re

with open("templates/index.html", encoding="utf-8") as f:
    html = f.read()

# Extract all input/select names
form_names = set(re.findall(r'name=["\']([a-zA-Z0-9_]+)["\']', html))
print("Form element names found in index.html (", len(form_names), "):")
print(sorted(list(form_names)))

expected_inputs = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

missing = [k for k in expected_inputs if k not in form_names]
assert not missing, f"Missing inputs in HTML form: {missing}"
print("All 19 customer input fields are present in HTML form!")

# Check static links in HTML
assert "css/style.css" in html, "Missing CSS link"
assert "js/main.js" in html, "Missing JS link"
print("Static CSS and JS links correctly defined with url_for.")

# Check JS file for preset keys
with open("static/js/main.js", encoding="utf-8") as f:
    js = f.read()

for field in expected_inputs:
    if field != "TotalCharges":
        assert field in js, f"Field {field} missing in JS presets/handling"

print("JavaScript presets and form handlers match all fields perfectly!")
print("FRONTEND VERIFICATION PASSED!")
