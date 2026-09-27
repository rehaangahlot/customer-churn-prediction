# ⚡ Customer Churn Prediction Web Application (Deep Learning ANN)

A production-ready, multi-device Customer Churn Prediction web application powered by an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras**, served via **Flask**, and styled with modern, responsive **HTML5 / CSS3 / JavaScript**.

Based on the [Codebasics Telco Customer Churn Deep Learning Tutorial](https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/11_chrun_prediction/churn.ipynb).

---

## 🌟 Key Highlights & Architecture

- **End-to-End Deep Learning**: 3-layer ANN trained on the Telco Customer Churn dataset with exact preprocessing, one-hot encoding, and MinMax scaling.
- **RESTful Prediction API**: High-performance `POST /predict` and `GET /health` endpoints.
- **Multi-Device Responsive Interface**: Works seamlessly on mobile phones (iOS & Android), tablets, laptops, and desktop computers.
- **1-Click Testing Profiles**: Instant pre-fills for High-Risk, Loyal, and Moderate customer profiles for rapid testing without manual typing.
- **Production & Cloud Ready**: Zero hardcoded local Windows paths, environment-driven port handling, container/cloud-ready with `Procfile` and `gunicorn`.

```
[ Browser on Phone / PC / Mac ]
               │
      (HTTPS / HTTP Request)
               ▼
[ Responsive Frontend (HTML/CSS/JS) ]
               │
         (POST /predict)
               ▼
   [ Flask Backend (app.py) ]
               │
    (MinMaxScaler Transform)
               ▼
 [ TensorFlow / Keras ANN Model ]
               │
(Churn Probability & Risk Analysis)
               ▼
       [ JSON Response ]
```

---

## 📁 Project Structure

```
churn-prediction/
│
├── data/
│   └── customer_churn.csv        # Telco Customer Churn dataset (7,043 rows)
│
├── model/
│   ├── churn_model.keras         # Saved TensorFlow / Keras ANN model
│   ├── scaler.joblib             # Fitted MinMaxScaler for numeric features
│   └── metadata.json             # Model metrics, architecture & feature schema
│
├── reports/
│   └── confusion_matrix.png      # High-resolution confusion matrix plot
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py          # Data cleaning, encoding & feature transformation
│
├── static/
│   ├── css/
│   │   └── style.css             # Responsive layout (Desktop, Tablet, Mobile)
│   └── js/
│       └── main.js               # Client-side validation, presets & async API calls
│
├── templates/
│   └── index.html                # Modern dashboard UI
│
├── train.py                      # Standalone ANN training & evaluation script
├── app.py                        # Production Flask application & REST API
├── requirements.txt              # Production dependency specifications
├── Procfile                      # Cloud WSGI process declaration (Render/Railway)
├── runtime.txt                   # Python runtime pin (Python 3.12)
├── .gitignore                    # Git exclusion rules
└── README.md                     # Comprehensive documentation & deployment guide
```

---

## 🚀 Local Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your machine.
- Git (optional, for version control).

### Step 1: Open Terminal / PowerShell
Clone or navigate to the project directory:
```bash
cd "f:\churn prediction"
```

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Instructions

The project includes pre-trained model artifacts in `model/`. If you want to re-train the model from scratch:

```bash
python train.py
```

### What `train.py` Does:
1. Automatically verifies or downloads `data/customer_churn.csv`.
2. Cleans blank `TotalCharges` values and standardizes service columns.
3. Performs binary and one-hot encoding on categorical features.
4. Scales `tenure`, `MonthlyCharges`, and `TotalCharges` using `MinMaxScaler`.
5. Splits dataset into 80% train / 20% test (`random_state=5`).
6. Builds the ANN model:
   - **Input Layer**: 26 neurons (ReLU)
   - **Hidden Layer**: 15 neurons (ReLU)
   - **Output Layer**: 1 neuron (Sigmoid)
7. Trains for 100 epochs using `Adam` optimizer and `binary_crossentropy` loss.
8. Evaluates performance and generates:
   - **Accuracy**: ~77% - 80%
   - **Precision**: ~0.62
   - **Recall**: ~0.53
   - **F1-Score**: ~0.57
   - **Confusion Matrix**: Saved to `reports/confusion_matrix.png`
9. Exports `model/churn_model.keras`, `model/scaler.joblib`, and `model/metadata.json`.

---

## 🖥️ Local Testing Instructions

### Step 1: Start the Web Server
Run the Flask app:
```bash
python app.py
```

You should see output similar to:
```
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your-local-ip>:5000
```

### Step 2: Test in Your Browser
Open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

1. Notice the **Live Model Status** badge showing "AI Model Online".
2. Click **"🚨 High Risk Profile"** to instantly test a customer likely to churn (e.g., Month-to-Month, Fiber Optic, tenure=2).
3. Click **"🛡️ Loyal Customer Profile"** to test a stable customer (Two-Year contract, tenure=65).
4. Customize any values (tenure slider, monthly charges, contract) and click **"Predict Customer Churn"**.
5. Observe the animated probability progress bar, risk badge, and strategic retention recommendations.

### Step 3: Test Health Endpoint
Open in browser or terminal:
```bash
curl http://127.0.0.1:5000/health
```
Response:
```json
{
  "environment": "production",
  "ready": true,
  "status": "healthy",
  "timestamp": "2026-09-27T09:25:34.434965Z"
}
```

---

## 🌐 Deploying the Project Online

Deploy your application to a free/low-cost cloud platform so that **anyone can access it from any device (iPhone, Android, PC, Mac, Tablet) using a normal HTTPS link**, without needing Python or downloading any files.

We recommend **Render** or **Railway** for a student-friendly, zero-cost deployment workflow.

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com/).
2. Click the **"+"** icon in the top-right corner and select **"New repository"**.
3. Name your repository (e.g., `customer-churn-prediction-ann`).
4. Leave it as **Public** and do NOT initialize with README/license (we already have them).
5. Click **"Create repository"**.

### Step 2: Upload Your Project to GitHub
In your local project terminal, run the following commands:

```bash
git init
git add .
git commit -m "Initial commit: Complete Churn Prediction Web App with trained ANN model"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git push -u origin main
```

*(Ensure the `model/churn_model.keras` and `model/scaler.joblib` files are included in the commit so the cloud server has the pre-trained artifacts immediately ready).*

---

### Step 3: Deploy on Render (Recommended & Free)

1. Sign up or log in to **[Render.com](https://render.com/)** using your GitHub account.
2. In the Render Dashboard, click **"New +"** and choose **"Web Service"**.
3. Select **"Build and deploy from a Git repository"** and click **Next**.
4. Choose your `customer-churn-prediction-ann` repository and click **Connect**.
5. Configure the service settings:
   - **Name**: `churn-prediction-app` (or any unique name)
   - **Region**: Choose the region closest to you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Root Directory**: (Leave blank)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app
     ```
   - **Instance Type**: Select **Free**
6. Click **"Create Web Service"**.
7. Render will automatically clone your GitHub repository, install dependencies, load the model, and launch the Gunicorn WSGI server.
8. Once the build logs show `==> Your service is live 🎉`, your public HTTPS URL will appear at the top:
   👉 **`https://churn-prediction-app.onrender.com`**

---

### Alternative: Deploy on Railway
1. Go to **[Railway.app](https://railway.app/)** and sign in with GitHub.
2. Click **"New Project"** &rarr; **"Deploy from GitHub repo"**.
3. Select your repository.
4. Railway automatically detects `Procfile` and `requirements.txt`.
5. Under service settings, click **"Generate Domain"** to get your public HTTPS URL.

---

## 📱 Mobile & Multi-Device Testing Instructions

Once deployed online:

1. **On your Smartphone (iPhone / Android)**:
   - Open Safari, Chrome, or Firefox.
   - Enter your public HTTPS URL: `https://your-app-name.onrender.com`.
   - Verify that:
     - The page loads without horizontal scrollbars.
     - Form sections stack vertically and neatly.
     - Dropdowns and sliders are touch-friendly.
     - Tapping **"🚨 High Risk Profile"** automatically populates the form and generates the prediction card.
2. **On a Friend's Computer or Tablet**:
   - Share the link via WhatsApp, Email, or Slack.
   - They can use the full application immediately without installing Python, TensorFlow, or cloning the code.
3. **Verify Health Check Online**:
   - Visit `https://your-app-name.onrender.com/health` in any browser to verify `{"status": "healthy"}`.

---

## 🔌 Prediction REST API Reference

You can also integrate the deployed app with external systems, mobile apps, or scripts using the REST API.

### Endpoint: `POST /predict`
- **Content-Type**: `application/json`

#### Request Payload Example:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
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
  "MonthlyCharges": 89.50,
  "TotalCharges": 268.50
}
```

#### Response Example:
```json
{
  "churn_probability": 0.8842,
  "non_churn_probability": 0.1158,
  "prediction": "Churn",
  "risk_level": "High",
  "threshold": 0.5
}
```

#### cURL Example:
```bash
curl -X POST https://your-app-name.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":89.5,"TotalCharges":268.5}'
```

---

## 📊 Model Evaluation Details

| Metric | Result | Description |
|---|---|---|
| **Test Accuracy** | **76.9%** | Correct classifications on unseen test customers |
| **Precision (Churn=1)** | **0.62** | 62% of predicted churners actually churned |
| **Recall (Churn=1)** | **0.53** | 53% of actual churners correctly caught |
| **F1-Score (Churn=1)** | **0.57** | Harmonic balance between Precision & Recall |
| **ROC AUC** | **~0.83** | High discriminative power across threshold spectrum |

---

## 🛡️ License & Acknowledgements
- Reference Tutorial: [Codebasics Deep Learning Keras/TF Series](https://github.com/codebasics/deep-learning-keras-tf-tutorial)
- Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Built with Python, TensorFlow/Keras, Flask, Scikit-learn, and Modern CSS.
