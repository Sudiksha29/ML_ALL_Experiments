import time
import numpy as np
import pandas as pd
from services.model_registry import registry
from services.history_service import log_prediction

def predict_experiment(exp_id, switcher_id, form_data):
    """
    Dispatches prediction logic to the corresponding experiment model.
    Returns: dict with prediction, confidence, probability breakdown, explanation, execution_ms
    """
    start_time = time.time()
    result = {
        "status": "success",
        "exp_id": exp_id,
        "switcher_id": switcher_id,
        "prediction": None,
        "label": None,
        "confidence": None,
        "probabilities": {},
        "explanation": "",
        "execution_ms": 0
    }

    try:
        # EXP 02: Simple Linear Regression
        if exp_id == "exp02":
            variant = switcher_id or "placement"
            if variant == "placement":
                model = registry.get_model("exp02_slr_placement")
                cgpa = float(form_data.get("cgpa", 8.0))
                if cgpa < 0 or cgpa > 10:
                    raise ValueError("CGPA must be between 0.0 and 10.0")
                pred = float(model.predict([[cgpa]])[0])
                result["prediction"] = round(pred, 2)
                result["label"] = f"{pred:.2f} LPA"
                result["explanation"] = f"Using equation: Package = {model.coef_[0]:.4f} * {cgpa} + ({model.intercept_:.4f}). An increase of 1.0 CGPA increases expected package by {model.coef_[0]:.2f} LPA."
            else:
                model = registry.get_model("exp02_slr_hours")
                hours = float(form_data.get("hours", 5.0))
                pred = float(model.predict([[hours]])[0])
                result["prediction"] = round(pred, 1)
                result["label"] = f"{pred:.1f} Marks"
                result["explanation"] = f"Using equation: Score = {model.coef_[0]:.4f} * {hours} + ({model.intercept_:.4f}). Each study hour contributes approximately {model.coef_[0]:.1f} additional exam marks."

        # EXP 03: Multiple Linear Regression
        elif exp_id == "exp03":
            variant = switcher_id or "cgpa_iq"
            if variant == "cgpa_iq":
                model = registry.get_model("exp03_mlr_cgpa_iq")
                cgpa = float(form_data.get("cgpa", 8.0))
                iq = float(form_data.get("iq", 115.0))
                pred = float(model.predict([[cgpa, iq]])[0])
                result["prediction"] = round(pred, 2)
                result["label"] = f"{pred:.2f} LPA"
                result["explanation"] = f"Prediction: Package = {model.coef_[0]:.3f}*(CGPA={cgpa}) + {model.coef_[1]:.4f}*(IQ={iq}) + ({model.intercept_:.3f}). CGPA and IQ both positively contribute to placement package."
            else:
                model = registry.get_model("exp03_mlr_realestate")
                h_age = float(form_data.get("house_age", 15.0))
                dist = float(form_data.get("distance", 500.0))
                stores = float(form_data.get("convenience_stores", 5.0))
                lat = float(form_data.get("latitude", 24.98))
                lon = float(form_data.get("longitude", 121.54))
                pred = float(model.predict([[h_age, dist, stores, lat, lon]])[0])
                result["prediction"] = round(pred, 2)
                result["label"] = f"{pred:.2f} 10k NTD/ping"
                result["explanation"] = f"Unit area price estimated from 5 urban geographical and structural predictors. Distance to transit station has a strong negative correlation with property price."

        # EXP 04: Binary Logistic Regression
        elif exp_id == "exp04":
            variant = switcher_id or "titanic"
            if variant == "titanic":
                model = registry.get_model("exp04_lr_titanic")
                pclass = int(form_data.get("pclass", 1))
                sex = int(form_data.get("sex", 1)) # 0: male, 1: female
                age = float(form_data.get("age", 28.0))
                fare = float(form_data.get("fare", 50.0))
                X_in = pd.DataFrame([[pclass, sex, age, fare]], columns=['pclass', 'sex', 'age', 'fare'])
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                result["prediction"] = pred
                result["label"] = "Survived (1)" if pred == 1 else "Did Not Survive (0)"
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {
                    "Perished (0)": round(float(proba[0]) * 100, 1),
                    "Survived (1)": round(float(proba[1]) * 100, 1)
                }
                gender_str = "female" if sex == 1 else "male"
                result["explanation"] = f"For passenger ({gender_str}, class {pclass}, age {age}, fare ${fare:.1f}), the sigmoid probability of survival is {proba[1]*100:.1f}%. Historical survival was significantly higher for higher cabin classes and women."
            else:
                model = registry.get_model("exp04_lr_placement")
                cgpa = float(form_data.get("cgpa", 7.5))
                iq = float(form_data.get("iq", 110.0))
                X_in = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                result["prediction"] = pred
                result["label"] = "Placed (1)" if pred == 1 else "Not Placed (0)"
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {
                    "Not Placed (0)": round(float(proba[0]) * 100, 1),
                    "Placed (1)": round(float(proba[1]) * 100, 1)
                }
                result["explanation"] = f"With CGPA {cgpa} and IQ {iq}, the student has a {proba[1]*100:.1f}% probability of securing campus placement based on the logistic decision threshold."

        # EXP 05: Multiclass Logistic Regression
        elif exp_id == "exp05":
            variant = switcher_id or "student"
            if variant == "student":
                model = registry.get_model("exp05_multiclass_student")
                hours = float(form_data.get("study_hours", 6.0))
                att = float(form_data.get("attendance", 75.0))
                X_in = [[hours, att]]
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                class_names = ["Low Performance (0)", "Medium Performance (1)", "High Performance (2)"]
                result["prediction"] = pred
                result["label"] = class_names[pred]
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {class_names[i]: round(float(proba[i]) * 100, 1) for i in range(len(class_names))}
                result["explanation"] = f"Softmax multiclass distribution evaluated {hours} study hours and {att}% attendance. The model assigned highest log-odds probability ({proba[pred]*100:.1f}%) to {class_names[pred]}."
            else:
                model = registry.get_model("exp05_multiclass_iris")
                sl = float(form_data.get("sepal_length", 5.8))
                sw = float(form_data.get("sepal_width", 3.0))
                pl = float(form_data.get("petal_length", 4.2))
                pw = float(form_data.get("petal_width", 1.3))
                X_in = pd.DataFrame([[sl, sw, pl, pw]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                classes = ["Iris Setosa (0)", "Iris Versicolor (1)", "Iris Virginica (2)"]
                result["prediction"] = pred
                result["label"] = classes[pred]
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {classes[i]: round(float(proba[i]) * 100, 1) for i in range(3)}
                result["explanation"] = f"Morphological petal and sepal dimensions most strongly indicate {classes[pred]} with {proba[pred]*100:.1f}% confidence."

        # EXP 06: Decision Tree Classification (UCI Heart Disease)
        elif exp_id == "exp06":
            model = registry.get_model("exp06_heart")
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            vals = [float(form_data.get(fn, 0)) for fn in feature_names]
            X_in = pd.DataFrame([vals], columns=feature_names)
            pred = int(model.predict(X_in)[0])
            proba = model.predict_proba(X_in)[0]
            result["prediction"] = pred
            result["label"] = "Heart Disease Detected (1)" if pred == 1 else "No Heart Disease (0)"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                "No Disease (0)": round(float(proba[0]) * 100, 1),
                "Heart Disease Present (1)": round(float(proba[1]) * 100, 1)
            }
            top_factors = []
            if form_data.get('cp', 0) in [1, 2, 3]: top_factors.append("presence of typical/atypical chest pain")
            if float(form_data.get('thalach', 150)) < 130: top_factors.append("reduced maximum heart rate")
            if float(form_data.get('oldpeak', 0)) > 1.5: top_factors.append("elevated ST depression")
            factor_text = f" Driven primarily by {', '.join(top_factors)}." if top_factors else ""
            result["explanation"] = f"Decision tree traversed Gini-split test criteria.{factor_text} Note: Strictly for educational demonstration, not a medical diagnosis."

        # EXP 07: K-Nearest Neighbors (Diabetes & Breast Cancer)
        elif exp_id == "exp07":
            variant = switcher_id or "diabetes"
            if variant == "diabetes":
                model = registry.get_model("exp07_knn_diabetes")
                features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
                vals = [float(form_data.get(f, 0)) for f in features]
                X_in = pd.DataFrame([vals], columns=features)
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                result["prediction"] = pred
                result["label"] = "Diabetic (1)" if pred == 1 else "Non-Diabetic (0)"
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {
                    "Non-Diabetic (0)": round(float(proba[0]) * 100, 1),
                    "Diabetic (1)": round(float(proba[1]) * 100, 1)
                }
                result["explanation"] = f"Standardized patient features mapped against K=17 nearest neighbors in scaled Euclidean feature space. {proba[pred]*100:.1f}% of neighbor votes selected {result['label']}."
            else:
                model = registry.get_model("exp07_knn_breastcancer")
                features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
                # fallback for missing inputs
                defaults = [14.0, 19.0, 92.0, 650.0, 0.096, 0.104, 0.088, 0.048, 0.18, 0.062]
                vals = [float(form_data.get(f, d)) for f, d in zip(features, defaults)]
                X_in = pd.DataFrame([vals], columns=features)
                pred = int(model.predict(X_in)[0])
                proba = model.predict_proba(X_in)[0]
                result["prediction"] = pred
                result["label"] = "Malignant (1)" if pred == 1 else "Benign (0)"
                result["confidence"] = round(float(np.max(proba)) * 100, 1)
                result["probabilities"] = {
                    "Benign (0)": round(float(proba[0]) * 100, 1),
                    "Malignant (1)": round(float(proba[1]) * 100, 1)
                }
                result["explanation"] = f"Diagnostic nuclear measurements compared with K=9 nearest tissue biopsies in normalized metric space."

        # EXP 08: Gaussian Naive Bayes
        elif exp_id == "exp08":
            model = registry.get_model("exp08_banknote")
            features = ["variance", "skewness", "curtosis", "entropy"]
            vals = [float(form_data.get(f, 0.0)) for f in features]
            X_in = pd.DataFrame([vals], columns=features)
            pred = int(model.predict(X_in)[0])
            proba = model.predict_proba(X_in)[0]
            result["prediction"] = pred
            result["label"] = "Genuine Banknote (1)" if pred == 1 else "Counterfeit Banknote (0)"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                "Counterfeit (0)": round(float(proba[0]) * 100, 1),
                "Genuine (1)": round(float(proba[1]) * 100, 1)
            }
            result["explanation"] = f"Gaussian normal density curves computed independent likelihood for each wavelet feature. Class posterior probability P(C|X) yields {proba[pred]*100:.1f}% likelihood for {result['label']}."

        # EXP 09: Multinomial Naive Bayes (Campus Message Detective)
        elif exp_id == "exp09":
            variant = switcher_id or "teaching56"
            text_msg = form_data.get("message", "").strip()
            if not text_msg:
                text_msg = "Please submit your Machine Learning project report before 5 PM tomorrow."
            
            if variant == "teaching56":
                bundle = registry.get_model("exp09_mnb")
            else:
                bundle = registry.get_model("exp09_mnb_222")
            
            vectorizer = bundle["vectorizer"]
            model = bundle["model"]
            classes = bundle["classes"]

            bow = vectorizer.transform([text_msg])
            pred_class = model.predict(bow)[0]
            proba = model.predict_proba(bow)[0]

            matched_tokens = [w for w in text_msg.lower().split() if w in vectorizer.vocabulary_]

            result["prediction"] = str(pred_class)
            result["label"] = f"{pred_class.upper()} MESSAGE"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                str(classes[i]): round(float(proba[i]) * 100, 1) for i in range(len(classes))
            }
            result["explanation"] = f"Bag-of-Words transformation matched {len(matched_tokens)} key vocabulary terms: [{', '.join(matched_tokens[:6])}]. Applying Bayes' theorem with Laplace smoothing yielded {np.max(proba)*100:.1f}% posterior confidence."

        # EXP 10: Linear SVM
        elif exp_id == "exp10":
            model = registry.get_model("exp10_linear_svm")
            features = ["variance", "skewness", "curtosis", "entropy"]
            vals = [float(form_data.get(f, 0.0)) for f in features]
            X_in = pd.DataFrame([vals], columns=features)
            pred = int(model.predict(X_in)[0])
            proba = model.predict_proba(X_in)[0]
            result["prediction"] = pred
            result["label"] = "Genuine Banknote (1)" if pred == 1 else "Counterfeit Banknote (0)"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                "Counterfeit (0)": round(float(proba[0]) * 100, 1),
                "Genuine (1)": round(float(proba[1]) * 100, 1)
            }
            result["explanation"] = f"Input vector evaluated against optimal linear separating hyperplane (w·x + b = 0) with maximal margin. Prediction classified on positive/negative side of the boundary."

        # EXP 11: Non-Linear SVM (RBF)
        elif exp_id == "exp11":
            model = registry.get_model("exp11_nonlinear_svm")
            features = ["variance", "skewness", "curtosis", "entropy"]
            vals = [float(form_data.get(f, 0.0)) for f in features]
            X_in = pd.DataFrame([vals], columns=features)
            pred = int(model.predict(X_in)[0])
            proba = model.predict_proba(X_in)[0]
            result["prediction"] = pred
            result["label"] = "Genuine Banknote (1)" if pred == 1 else "Counterfeit Banknote (0)"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                "Counterfeit (0)": round(float(proba[0]) * 100, 1),
                "Genuine (1)": round(float(proba[1]) * 100, 1)
            }
            result["explanation"] = f"Radial Basis Function (RBF) kernel evaluated similarities exp(-gamma * ||x - x'||^2) with support vectors, synthesizing non-linear curved decision boundaries in original space."

        # EXP 12: Random Forest Classification
        elif exp_id == "exp12":
            model = registry.get_model("exp12_rf_diabetes")
            features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            vals = [float(form_data.get(f, 0.0)) for f in features]
            X_in = pd.DataFrame([vals], columns=features)
            pred = int(model.predict(X_in)[0])
            proba = model.predict_proba(X_in)[0]
            result["prediction"] = pred
            result["label"] = "Diabetic (1)" if pred == 1 else "Non-Diabetic (0)"
            result["confidence"] = round(float(np.max(proba)) * 100, 1)
            result["probabilities"] = {
                "Non-Diabetic (0)": round(float(proba[0]) * 100, 1),
                "Diabetic (1)": round(float(proba[1]) * 100, 1)
            }
            result["explanation"] = f"Ensemble of 100 bootstrapped decision trees aggregated votes. {proba[pred]*100:.1f}% of individual trees voted for {result['label']}, reducing single-tree variance."

        # EXP 13: Random Forest Regression
        elif exp_id == "exp13":
            model = registry.get_model("exp13_rf_glucose")
            features = ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            vals = [float(form_data.get(f, 0.0)) for f in features]
            X_in = pd.DataFrame([vals], columns=features)
            pred = float(model.predict(X_in)[0])
            result["prediction"] = round(pred, 1)
            result["label"] = f"{pred:.1f} mg/dL Glucose"
            result["explanation"] = f"Predicted continuous fasting plasma glucose by averaging individual numeric forecasts across 100 randomized decision trees."

        else:
            raise ValueError(f"Unknown experiment ID: {exp_id}")

        result["inputs"] = {k: v for k, v in form_data.items() if k != "switcher_id"}
        result["execution_ms"] = round((time.time() - start_time) * 1000, 2)
        # Log event to database
        log_prediction(
            exp_id=exp_id,
            switcher_id=switcher_id,
            inputs=dict(form_data),
            output=result["label"],
            confidence=result.get("confidence"),
            duration_ms=result["execution_ms"]
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["execution_ms"] = round((time.time() - start_time) * 1000, 2)

    return result
