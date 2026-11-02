/**
 * ML LabVerse — Prediction Engine & AJAX Submission Service
 */

function handlePredictionForm(formId, expId) {
    const form = document.getElementById(formId);
    if (!form) return;

    const resultBox = document.getElementById("predictionResultBox");
    const resultVal = document.getElementById("resultValue");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultExplanation = document.getElementById("resultExplanation");
    const resultBars = document.getElementById("resultProbBars");
    const submitBtn = form.querySelector("button[type='submit']");
    const originalBtnHtml = submitBtn ? submitBtn.innerHTML : "Predict";

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Show loading state
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner"></span> Running Model...`;
        }

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Check active switcher pill
        const activeSwitcher = form.closest(".card-panel")?.querySelector(".switcher-pill.active");
        const switcherId = activeSwitcher ? activeSwitcher.dataset.switcherId : form.querySelector("input[name='switcher_id']")?.value;
        if (switcherId) {
            data["switcher_id"] = switcherId;
        }

        try {
            const response = await fetch(`/api/predict/${expId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    switcher_id: switcherId,
                    inputs: data
                })
            });

            const res = await response.json();

            if (res.status === "success") {
                if (resultBox) resultBox.style.display = "block";
                if (resultVal) resultVal.textContent = res.label || res.prediction;
                
                if (resultConfidence) {
                    if (res.confidence !== null && res.confidence !== undefined) {
                        resultConfidence.textContent = `Model Confidence: ${res.confidence}% (Inference: ${res.execution_ms} ms)`;
                        resultConfidence.style.display = "block";
                    } else {
                        resultConfidence.textContent = `Inference Time: ${res.execution_ms} ms`;
                    }
                }

                if (resultExplanation) {
                    resultExplanation.textContent = res.explanation || "";
                }

                // Render Evaluated Features Tags
                const resultFeaturesList = document.getElementById("resultFeaturesList");
                const resultFeaturesBox = document.getElementById("resultFeaturesBox");
                const chartPlottedBadge = document.getElementById("chartPlottedBadge");
                if (resultFeaturesList) {
                    resultFeaturesList.innerHTML = "";
                    const inputsToDisplay = res.inputs || data;
                    const friendlyNames = {
                        "variance": "Wavelet Variance (X₁)",
                        "skewness": "Wavelet Skewness (X₂)",
                        "curtosis": "Wavelet Curtosis (X₃)",
                        "entropy": "Image Entropy (X₄)",
                        "cgpa": "CGPA",
                        "iq": "IQ Score",
                        "hours": "Study Hours / Day",
                        "study_hours": "Daily Study Hours",
                        "attendance": "Attendance (%)",
                        "age": "Patient / Passenger Age",
                        "fare": "Ticket Fare ($)",
                        "pclass": "Cabin Class",
                        "sex": "Sex",
                        "cp": "Chest Pain Type",
                        "trestbps": "Resting Blood Pressure",
                        "chol": "Serum Cholesterol",
                        "fbs": "Fasting Blood Sugar",
                        "restecg": "Resting ECG",
                        "thalach": "Max Heart Rate",
                        "exang": "Exercise Angina",
                        "oldpeak": "ST Depression",
                        "slope": "ST Slope",
                        "ca": "Major Vessels",
                        "thal": "Thalassemia",
                        "Pregnancies": "Pregnancies",
                        "Glucose": "Fasting Plasma Glucose",
                        "BloodPressure": "Diastolic Blood Pressure",
                        "SkinThickness": "Triceps Skin Fold",
                        "Insulin": "2-Hour Serum Insulin",
                        "BMI": "Body Mass Index (BMI)",
                        "DiabetesPedigreeFunction": "Diabetes Pedigree Score",
                        "message": "Campus Text Message",
                        "house_age": "House Age (years)",
                        "distance": "Distance to MRT (m)",
                        "convenience_stores": "Convenience Stores Count",
                        "latitude": "Latitude",
                        "longitude": "Longitude"
                    };

                    for (const [k, v] of Object.entries(inputsToDisplay)) {
                        if (k === "switcher_id") continue;
                        const pill = document.createElement("div");
                        pill.className = "feature-input-pill";
                        const labelName = friendlyNames[k] || k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                        pill.innerHTML = `<span class="pill-label">${labelName}:</span> <strong class="pill-val">${v}</strong>`;
                        resultFeaturesList.appendChild(pill);
                    }

                    if (chartPlottedBadge) {
                        chartPlottedBadge.style.display = ["exp02", "exp10", "exp11"].includes(expId) ? "inline-flex" : "none";
                    }
                    if (resultFeaturesBox) resultFeaturesBox.style.display = "block";
                }

                // Render probability distribution bars if available
                if (resultBars) {
                    resultBars.innerHTML = "";
                    if (res.probabilities && Object.keys(res.probabilities).length > 0) {
                        for (const [cls, prob] of Object.entries(res.probabilities)) {
                            const row = document.createElement("div");
                            row.className = "meter-row";
                            row.innerHTML = `
                                <div class="meter-header">
                                    <span>${cls}</span>
                                    <span>${prob}%</span>
                                </div>
                                <div class="meter-track">
                                    <div class="meter-fill" style="width: ${prob}%"></div>
                                </div>
                            `;
                            resultBars.appendChild(row);
                        }
                    }
                }

                // Dispatch event so active chart can immediately plot the predicted feature point
                const event = new CustomEvent("ml:prediction:success", {
                    detail: {
                        expId: expId,
                        switcherId: switcherId,
                        inputs: res.inputs || data,
                        result: res
                    }
                });
                window.dispatchEvent(event);

                // Scroll smoothly to result
                resultBox?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            } else {
                alert(`Prediction Error: ${res.error || "Unknown server error"}`);
            }
        } catch (err) {
            console.error("Prediction error:", err);
            alert("Error communicating with ML inference service: " + err.message);
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHtml;
            }
        }
    });
}

/**
 * Loads pre-filled realistic sample data for instant testing
 */
async function loadSamplePreset(expId, switcherId) {
    try {
        const url = `/api/sample/${expId}${switcherId ? '?switcher_id=' + switcherId : ''}`;
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === "success" && data.sample) {
            const form = document.querySelector(".experiment-form");
            if (!form) return;

            for (const [key, val] of Object.entries(data.sample)) {
                const input = form.querySelector(`[name="${key}"]`);
                if (input) {
                    input.value = val;
                    // Trigger input event for interactive feedback
                    input.dispatchEvent(new Event('input'));
                }
            }
        }
    } catch (e) {
        console.error("Error loading sample preset:", e);
    }
}
