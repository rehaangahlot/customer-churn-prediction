/**
 * Customer Churn Predictor Frontend Logic.
 * Handles interactive form presets, real-time recalculations,
 * API requests to /predict, and dynamic gauge updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("churnForm");
    const tenureSlider = document.getElementById("tenure");
    const tenureVal = document.getElementById("tenureVal");
    const monthlyChargesInput = document.getElementById("MonthlyCharges");
    const totalChargesInput = document.getElementById("TotalCharges");
    const btnSubmit = document.getElementById("btnSubmit");
    const btnText = btnSubmit.querySelector(".btn-text");
    const btnSpinner = document.getElementById("btnSpinner");

    // Result Card Elements
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const predictionBadge = document.getElementById("predictionBadge");
    const riskBadge = document.getElementById("riskBadge");
    const churnPercentage = document.getElementById("churnPercentage");
    const retentionPercentage = document.getElementById("retentionPercentage");
    const progressBar = document.getElementById("progressBar");
    const adviceTitle = document.getElementById("adviceTitle");
    const adviceText = document.getElementById("adviceText");
    const factorsList = document.getElementById("factorsList");
    const jsonResponse = document.getElementById("jsonResponse");

    // Status Indicator
    const healthIndicator = document.getElementById("healthIndicator");
    const healthStatus = document.getElementById("healthStatus");

    // 1. Check System Health
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (res.ok && data.status === "healthy") {
                healthIndicator.className = "status-indicator online";
                healthStatus.textContent = "AI Model Online";
            } else {
                healthIndicator.className = "status-indicator offline";
                healthStatus.textContent = "Model Degraded";
            }
        } catch (e) {
            healthIndicator.className = "status-indicator offline";
            healthStatus.textContent = "Server Offline";
        }
    }
    checkHealth();

    // 2. Real-time Tenure & TotalCharges estimation
    function autoCalcTotal() {
        const tenure = parseFloat(tenureSlider.value) || 0;
        const monthly = parseFloat(monthlyChargesInput.value) || 0;
        tenureVal.textContent = tenure;
        totalChargesInput.value = (tenure * monthly).toFixed(2);
    }

    tenureSlider.addEventListener("input", autoCalcTotal);
    monthlyChargesInput.addEventListener("input", autoCalcTotal);

    // 3. Quick Test Presets
    const presets = {
        highRisk: {
            gender: "Female",
            SeniorCitizen: "0",
            Partner: "No",
            Dependents: "No",
            tenure: 2,
            PhoneService: "Yes",
            MultipleLines: "No",
            InternetService: "Fiber optic",
            OnlineSecurity: "No",
            OnlineBackup: "No",
            DeviceProtection: "No",
            TechSupport: "No",
            StreamingTV: "Yes",
            StreamingMovies: "Yes",
            Contract: "Month-to-month",
            PaperlessBilling: "Yes",
            PaymentMethod: "Electronic check",
            MonthlyCharges: 95.50
        },
        loyal: {
            gender: "Male",
            SeniorCitizen: "0",
            Partner: "Yes",
            Dependents: "Yes",
            tenure: 65,
            PhoneService: "Yes",
            MultipleLines: "Yes",
            InternetService: "DSL",
            OnlineSecurity: "Yes",
            OnlineBackup: "Yes",
            DeviceProtection: "Yes",
            TechSupport: "Yes",
            StreamingTV: "No",
            StreamingMovies: "No",
            Contract: "Two year",
            PaperlessBilling: "No",
            PaymentMethod: "Credit card (automatic)",
            MonthlyCharges: 58.00
        },
        moderate: {
            gender: "Female",
            SeniorCitizen: "0",
            Partner: "Yes",
            Dependents: "No",
            tenure: 24,
            PhoneService: "Yes",
            MultipleLines: "Yes",
            InternetService: "Fiber optic",
            OnlineSecurity: "No",
            OnlineBackup: "Yes",
            DeviceProtection: "No",
            TechSupport: "No",
            StreamingTV: "Yes",
            StreamingMovies: "No",
            Contract: "One year",
            PaperlessBilling: "Yes",
            PaymentMethod: "Bank transfer (automatic)",
            MonthlyCharges: 76.50
        }
    };

    function applyPreset(data) {
        for (const [key, val] of Object.entries(data)) {
            const field = form.elements[key];
            if (field) {
                field.value = val;
            }
        }
        autoCalcTotal();
    }

    document.getElementById("btnPresetHighRisk").addEventListener("click", () => {
        applyPreset(presets.highRisk);
        form.requestSubmit();
    });

    document.getElementById("btnPresetLoyal").addEventListener("click", () => {
        applyPreset(presets.loyal);
        form.requestSubmit();
    });

    document.getElementById("btnPresetModerate").addEventListener("click", () => {
        applyPreset(presets.moderate);
        form.requestSubmit();
    });

    document.getElementById("btnClearForm").addEventListener("click", () => {
        form.reset();
        autoCalcTotal();
        resultContent.style.display = "none";
        resultPlaceholder.style.display = "block";
    });

    // 4. Form Submission & Prediction API Call
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Loading UI state
        btnSubmit.disabled = true;
        btnText.textContent = "Analyzing Customer...";
        btnSpinner.style.display = "inline-block";

        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => {
            payload[key] = value;
        });

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction request failed");
            }

            renderPrediction(result, payload);

            // On mobile viewports, scroll smoothly to the result card
            if (window.innerWidth < 992) {
                document.getElementById("resultCard").scrollIntoView({ behavior: "smooth" });
            }

        } catch (err) {
            alert(`Error making prediction: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnText.textContent = "Predict Customer Churn";
            btnSpinner.style.display = "none";
        }
    });

    // 5. Render Results to UI
    function renderPrediction(result, inputData) {
        resultPlaceholder.style.display = "none";
        resultContent.style.display = "block";

        const churnProb = result.churn_probability;
        const nonChurnProb = result.non_churn_probability;
        const isChurn = result.prediction === "Churn";
        const riskLevel = result.risk_level;

        // Prediction Badge
        predictionBadge.textContent = isChurn ? "Likely to Churn" : "Will Retain";
        predictionBadge.className = `prediction-badge ${isChurn ? "churn-yes" : "churn-no"}`;

        // Risk Level Badge
        riskBadge.textContent = `${riskLevel} Risk`;
        riskBadge.className = `risk-badge risk-${riskLevel.toLowerCase()}`;

        // Gauge percentage
        const churnPctFormatted = `${(churnProb * 100).toFixed(1)}%`;
        const retainPctFormatted = `${(nonChurnProb * 100).toFixed(1)}%`;
        churnPercentage.textContent = churnPctFormatted;
        retentionPercentage.textContent = retainPctFormatted;

        // Progress Bar
        progressBar.style.width = churnPctFormatted;
        if (churnProb >= 0.7) {
            progressBar.style.backgroundColor = "var(--danger)";
        } else if (churnProb >= 0.4) {
            progressBar.style.backgroundColor = "var(--warning)";
        } else {
            progressBar.style.backgroundColor = "var(--success)";
        }

        // Actionable Strategic Advice
        if (isChurn) {
            adviceTitle.textContent = "⚠️ High Churn Risk: Urgent Retention Actions";
            adviceText.textContent = `Customer has a ${churnPctFormatted} probability of leaving. Key risk factors: ${inputData.Contract} contract, payment via ${inputData.PaymentMethod}, and high charges ($${inputData.MonthlyCharges}/mo). Recommend offering a 1-year contract discount, bundling Free Tech Support, or switching to automatic billing.`;
        } else {
            adviceTitle.textContent = "✅ Customer Loyal & Stable";
            adviceText.textContent = `Customer has a ${retainPctFormatted} retention probability. Tenure (${inputData.tenure} months) and contract security (${inputData.Contract}) indicate strong customer loyalty. Maintain service quality and periodically reward with loyalty perks.`;
        }

        // Factors List
        factorsList.innerHTML = `
            <li>Contract: <strong>${inputData.Contract}</strong></li>
            <li>Internet: <strong>${inputData.InternetService}</strong></li>
            <li>Tenure: <strong>${inputData.tenure} mo</strong></li>
            <li>Monthly: <strong>$${inputData.MonthlyCharges}</strong></li>
            <li>Payment: <strong>${inputData.PaymentMethod}</strong></li>
            <li>Paperless: <strong>${inputData.PaperlessBilling}</strong></li>
            <li>Tech Support: <strong>${inputData.TechSupport}</strong></li>
        `;

        // JSON Response Dump
        jsonResponse.textContent = JSON.stringify(result, null, 2);
    }
});
