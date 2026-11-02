import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.datasets import load_iris

import config

def train_all():
    metrics = {}
    print("--- Starting ML LabVerse Training Pipeline ---")

    # -------------------------------------------------------------
    # EXP 02: Simple Linear Regression
    # -------------------------------------------------------------
    # 1. Placement Dataset
    placement_csv = os.path.join(config.DATA_RAW_DIR, "Placement.csv")
    df_p = pd.read_csv(placement_csv)
    # clean columns if needed
    col_x = [c for c in df_p.columns if "cgpa" in c.lower()][0]
    col_y = [c for c in df_p.columns if "package" in c.lower()][0]
    X_p = df_p[[col_x]]
    y_p = df_p[col_y]
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=2)
    slr_placement = LinearRegression()
    slr_placement.fit(X_train_p, y_train_p)
    y_pred_p = slr_placement.predict(X_test_p)

    joblib.dump(slr_placement, os.path.join(config.MODELS_DIR, "exp02_slr_placement.joblib"))

    # 2. Study Hours Dataset (Notebook Part 1)
    df_hours = pd.DataFrame({
        "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "Score": [35, 40, 50, 55, 65, 70, 75, 85]
    })
    slr_hours = LinearRegression()
    slr_hours.fit(df_hours[["Hours"]], df_hours["Score"])
    y_pred_h = slr_hours.predict(df_hours[["Hours"]])

    joblib.dump(slr_hours, os.path.join(config.MODELS_DIR, "exp02_slr_hours.joblib"))

    metrics["exp02"] = {
        "placement": {
            "slope": float(slr_placement.coef_[0]),
            "intercept": float(slr_placement.intercept_),
            "equation": f"Package = {slr_placement.coef_[0]:.4f} * CGPA + ({slr_placement.intercept_:.4f})",
            "mae": float(mean_absolute_error(y_test_p, y_pred_p)),
            "mse": float(mean_squared_error(y_test_p, y_pred_p)),
            "r2": float(r2_score(y_test_p, y_pred_p)),
            "train_samples": len(X_train_p),
            "test_samples": len(X_test_p),
            "scatter_x": df_p[col_x].tolist(),
            "scatter_y": df_p[col_y].tolist(),
            "line_x": [float(df_p[col_x].min()), float(df_p[col_x].max())],
            "line_y": [float(slr_placement.predict([[df_p[col_x].min()]])[0]), float(slr_placement.predict([[df_p[col_x].max()]])[0])]
        },
        "hours": {
            "slope": float(slr_hours.coef_[0]),
            "intercept": float(slr_hours.intercept_),
            "equation": f"Score = {slr_hours.coef_[0]:.4f} * Hours + ({slr_hours.intercept_:.4f})",
            "mae": float(mean_absolute_error(df_hours["Score"], y_pred_h)),
            "mse": float(mean_squared_error(df_hours["Score"], y_pred_h)),
            "r2": float(r2_score(df_hours["Score"], y_pred_h)),
            "train_samples": len(df_hours),
            "test_samples": len(df_hours),
            "scatter_x": df_hours["Hours"].tolist(),
            "scatter_y": df_hours["Score"].tolist(),
            "line_x": [1.0, 8.0],
            "line_y": [float(slr_hours.predict([[1]])[0]), float(slr_hours.predict([[8]])[0])]
        }
    }
    print("✓ Exp 02 trained and saved")

    # -------------------------------------------------------------
    # EXP 03: Multiple Linear Regression
    # -------------------------------------------------------------
    # 1. CGPA + IQ -> Package
    df_cgpa_iq = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "Cgpa_iq_package.csv"))
    # column detection
    cols = [c.strip().lower() for c in df_cgpa_iq.columns]
    df_cgpa_iq.columns = cols
    # drop Unnamed if present
    df_cgpa_iq = df_cgpa_iq[[c for c in df_cgpa_iq.columns if "unnamed" not in c]]
    X_mlr = df_cgpa_iq[["cgpa", "iq"]]
    y_mlr = df_cgpa_iq["package"]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mlr, y_mlr, test_size=0.2, random_state=42)
    mlr_model = LinearRegression()
    mlr_model.fit(X_train_m, y_train_m)
    y_pred_m = mlr_model.predict(X_test_m)

    joblib.dump(mlr_model, os.path.join(config.MODELS_DIR, "exp03_mlr_cgpa_iq.joblib"))

    # 2. Real Estate Price Prediction
    re_path = os.path.join(config.DATA_RAW_DIR, "Real estate - Real estate.csv")
    df_re = pd.read_csv(re_path)
    re_features = ['house age', 'distance to the nearest station', 'number of convenience stores', 'latitude', 'longitude']
    re_target = 'house price of unit area'
    X_re = df_re[re_features]
    y_re = df_re[re_target]
    X_train_re, X_test_re, y_train_re, y_test_re = train_test_split(X_re, y_re, test_size=0.2, random_state=42)
    mlr_re = LinearRegression()
    mlr_re.fit(X_train_re, y_train_re)
    y_pred_re = mlr_re.predict(X_test_re)

    joblib.dump(mlr_re, os.path.join(config.MODELS_DIR, "exp03_mlr_realestate.joblib"))

    metrics["exp03"] = {
        "cgpa_iq": {
            "features": ["CGPA", "IQ"],
            "coefficients": [float(c) for c in mlr_model.coef_],
            "intercept": float(mlr_model.intercept_),
            "equation": f"Package = {mlr_model.coef_[0]:.3f}*(CGPA) + {mlr_model.coef_[1]:.4f}*(IQ) + ({mlr_model.intercept_:.3f})",
            "mse": float(mean_squared_error(y_test_m, y_pred_m)),
            "mae": float(mean_absolute_error(y_test_m, y_pred_m)),
            "r2": float(r2_score(y_test_m, y_pred_m)),
            "actual": [round(float(v), 2) for v in y_test_m.iloc[:15].tolist()],
            "predicted": [round(float(v), 2) for v in y_pred_m[:15].tolist()]
        },
        "realestate": {
            "features": re_features,
            "coefficients": [float(c) for c in mlr_re.coef_],
            "intercept": float(mlr_re.intercept_),
            "mse": float(mean_squared_error(y_test_re, y_pred_re)),
            "mae": float(mean_absolute_error(y_test_re, y_pred_re)),
            "r2": float(r2_score(y_test_re, y_pred_re)),
            "actual": [round(float(v), 2) for v in y_test_re.iloc[:15].tolist()],
            "predicted": [round(float(v), 2) for v in y_pred_re[:15].tolist()]
        }
    }
    print("✓ Exp 03 trained and saved")

    # -------------------------------------------------------------
    # EXP 04: Binary Logistic Regression
    # -------------------------------------------------------------
    # 1. Titanic Dataset
    titanic_path = os.path.join(config.DATA_RAW_DIR, "titanic.csv")
    df_titanic = pd.read_csv(titanic_path)
    # columns: survived, pclass, sex, age, fare
    df_t = df_titanic[['survived', 'pclass', 'sex', 'age', 'fare']].copy()
    if df_t['sex'].dtype == object:
        df_t['sex'] = df_t['sex'].map({'male': 0, 'female': 1})
    df_t['age'] = df_t['age'].fillna(df_t['age'].median())
    df_t['fare'] = df_t['fare'].fillna(df_t['fare'].median())

    X_t = df_t[['pclass', 'sex', 'age', 'fare']]
    y_t = df_t['survived']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)
    
    pipe_t = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    pipe_t.fit(X_train_t, y_train_t)
    y_pred_t = pipe_t.predict(X_test_t)
    y_proba_t = pipe_t.predict_proba(X_test_t)[:, 1]

    joblib.dump(pipe_t, os.path.join(config.MODELS_DIR, "exp04_lr_titanic.joblib"))

    # 2. Placement Binary Dataset
    p_yn_path = os.path.join(config.DATA_RAW_DIR, "Placement_Y_N.csv")
    df_pyn = pd.read_csv(p_yn_path)
    pyn_cols = [c.strip().lower() for c in df_pyn.columns]
    df_pyn.columns = pyn_cols
    df_pyn = df_pyn[[c for c in df_pyn.columns if "unnamed" not in c]]
    X_pyn = df_pyn[['cgpa', 'iq']]
    y_pyn = df_pyn['placement']
    X_train_pyn, X_test_pyn, y_train_pyn, y_test_pyn = train_test_split(X_pyn, y_pyn, test_size=0.2, random_state=42, stratify=y_pyn)
    pipe_pyn = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    pipe_pyn.fit(X_train_pyn, y_train_pyn)
    y_pred_pyn = pipe_pyn.predict(X_test_pyn)
    y_proba_pyn = pipe_pyn.predict_proba(X_test_pyn)[:, 1]

    joblib.dump(pipe_pyn, os.path.join(config.MODELS_DIR, "exp04_lr_placement.joblib"))

    fpr_t, tpr_t, _ = roc_curve(y_test_t, y_proba_t)
    metrics["exp04"] = {
        "titanic": {
            "accuracy": float(accuracy_score(y_test_t, y_pred_t)),
            "precision": float(precision_score(y_test_t, y_pred_t)),
            "recall": float(recall_score(y_test_t, y_pred_t)),
            "f1": float(f1_score(y_test_t, y_pred_t)),
            "auc": float(roc_auc_score(y_test_t, y_proba_t)),
            "confusion_matrix": confusion_matrix(y_test_t, y_pred_t).tolist(),
            "classes": ["Perished (0)", "Survived (1)"],
            "roc_fpr": [round(float(x), 3) for x in fpr_t[::max(1, len(fpr_t)//20)]],
            "roc_tpr": [round(float(y), 3) for y in tpr_t[::max(1, len(tpr_t)//20)]],
            "features": ['pclass', 'sex', 'age', 'fare']
        },
        "placement": {
            "accuracy": float(accuracy_score(y_test_pyn, y_pred_pyn)),
            "precision": float(precision_score(y_test_pyn, y_pred_pyn)),
            "recall": float(recall_score(y_test_pyn, y_pred_pyn)),
            "f1": float(f1_score(y_test_pyn, y_pred_pyn)),
            "auc": float(roc_auc_score(y_test_pyn, y_proba_pyn)),
            "confusion_matrix": confusion_matrix(y_test_pyn, y_pred_pyn).tolist(),
            "classes": ["Not Placed (0)", "Placed (1)"],
            "features": ['cgpa', 'iq']
        }
    }
    print("✓ Exp 04 trained and saved")

    # -------------------------------------------------------------
    # EXP 05: Multiclass Logistic Regression
    # -------------------------------------------------------------
    # 1. Student Performance (10 instances from notebook)
    df_perf = pd.DataFrame({
        'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'Performance': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    })
    X_perf = df_perf[['Study_Hours', 'Attendance']]
    y_perf = df_perf['Performance']
    # Fit full dataset or test split as in notebook
    perf_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    perf_model.fit(X_perf, y_perf)

    joblib.dump(perf_model, os.path.join(config.MODELS_DIR, "exp05_multiclass_student.joblib"))

    # 2. Iris Flower Species
    iris = load_iris()
    X_iris = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y_iris = iris.target
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)
    iris_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    iris_model.fit(X_train_i, y_train_i)
    y_pred_i = iris_model.predict(X_test_i)

    joblib.dump(iris_model, os.path.join(config.MODELS_DIR, "exp05_multiclass_iris.joblib"))

    cm_i = confusion_matrix(y_test_i, y_pred_i).tolist()
    metrics["exp05"] = {
        "student": {
            "accuracy": float(perf_model.score(X_perf, y_perf)),
            "classes": ["0: Low Performance", "1: Medium Performance", "2: High Performance"],
            "features": ["Study_Hours", "Attendance"],
            "sample_predictions": [
                {"study_hours": 3, "attendance": 60, "pred": "Low (0)"},
                {"study_hours": 5, "attendance": 70, "pred": "Medium (1)"},
                {"study_hours": 9, "attendance": 90, "pred": "High (2)"}
            ]
        },
        "iris": {
            "accuracy": float(accuracy_score(y_test_i, y_pred_i)),
            "confusion_matrix": cm_i,
            "classes": ["Setosa (0)", "Versicolor (1)", "Virginica (2)"],
            "features": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            "macro_f1": float(f1_score(y_test_i, y_pred_i, average='macro'))
        }
    }
    print("✓ Exp 05 trained and saved")

    # -------------------------------------------------------------
    # EXP 06: Decision Tree Classification (UCI Heart Disease)
    # -------------------------------------------------------------
    heart_path = os.path.join(config.DATA_RAW_DIR, "heart.csv")
    df_heart = pd.read_csv(heart_path)
    X_heart = df_heart.drop('target', axis=1)
    y_heart = df_heart['target']
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart)
    
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    dt_model.fit(X_train_h, y_train_h)
    y_pred_h = dt_model.predict(X_test_h)
    y_proba_h = dt_model.predict_proba(X_test_h)[:, 1]

    joblib.dump(dt_model, os.path.join(config.MODELS_DIR, "exp06_dt_heart.joblib"))

    feat_importances = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(X_heart.columns, dt_model.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )
    fpr_h, tpr_h, _ = roc_curve(y_test_h, y_proba_h)
    tree_rules = export_text(dt_model, feature_names=list(X_heart.columns))

    metrics["exp06"] = {
        "accuracy": float(accuracy_score(y_test_h, y_pred_h)),
        "precision": float(precision_score(y_test_h, y_pred_h)),
        "recall": float(recall_score(y_test_h, y_pred_h)),
        "f1": float(f1_score(y_test_h, y_pred_h)),
        "auc": float(roc_auc_score(y_test_h, y_proba_h)),
        "confusion_matrix": confusion_matrix(y_test_h, y_pred_h).tolist(),
        "classes": ["No Heart Disease (0)", "Heart Disease Present (1)"],
        "feature_importances": feat_importances,
        "tree_rules_text": tree_rules[:1200],
        "features": list(X_heart.columns),
        "roc_fpr": [round(float(x), 3) for x in fpr_h[::max(1, len(fpr_h)//15)]],
        "roc_tpr": [round(float(y), 3) for y in tpr_h[::max(1, len(tpr_h)//15)]],
        "train_samples": len(X_train_h),
        "test_samples": len(X_test_h)
    }
    print("✓ Exp 06 trained and saved")

    # -------------------------------------------------------------
    # EXP 07: K-Nearest Neighbors (KNN) - Diabetes & Breast Cancer
    # -------------------------------------------------------------
    # 1. Diabetes Dataset
    diab_path = os.path.join(config.DATA_RAW_DIR, "diabetes.csv")
    df_diab = pd.read_csv(diab_path)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_diab_clean = df_diab.copy()
    df_diab_clean[zero_cols] = df_diab_clean[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        df_diab_clean[col] = df_diab_clean.groupby("Outcome")[col].transform(lambda s: s.fillna(s.median()))

    X_diab = df_diab_clean.drop(columns=["Outcome"])
    y_diab = df_diab_clean["Outcome"]
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab)

    # Cross-validation tune K
    scaler_d = StandardScaler()
    X_train_d_sc = scaler_d.fit_transform(X_train_d)
    X_test_d_sc = scaler_d.transform(X_test_d)

    k_range = list(range(1, 31, 2))
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_d_sc, y_train_d, cv=10, scoring="accuracy")
        cv_scores.append(float(scores.mean()))
    best_k = k_range[int(np.argmax(cv_scores))]

    knn_diab = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2))
    ])
    knn_diab.fit(X_train_d, y_train_d)
    y_pred_d = knn_diab.predict(X_test_d)
    y_proba_d = knn_diab.predict_proba(X_test_d)[:, 1]

    joblib.dump(knn_diab, os.path.join(config.MODELS_DIR, "exp07_knn_diabetes.joblib"))

    # 2. Breast Cancer Dataset
    bc_path = os.path.join(config.DATA_RAW_DIR, "breast-cancer.csv")
    df_bc = pd.read_csv(bc_path)
    # clean diagnosis column
    target_col = [c for c in df_bc.columns if "diagnosis" in c.lower()][0]
    y_bc = df_bc[target_col].map({'M': 1, 'B': 0, 1: 1, 0: 0})
    ignore_cols = [c for c in df_bc.columns if "id" in c.lower() or "unnamed" in c.lower() or c == target_col]
    X_bc = df_bc.drop(columns=ignore_cols)
    # select the 10 mean features for easy web form
    mean_cols = [c for c in X_bc.columns if "mean" in c.lower()]
    if not mean_cols:
        mean_cols = list(X_bc.columns[:10])
    X_bc_mean = X_bc[mean_cols]
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc_mean, y_bc, test_size=0.2, random_state=42, stratify=y_bc)
    knn_bc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=9))
    ])
    knn_bc.fit(X_train_bc, y_train_bc)
    y_pred_bc = knn_bc.predict(X_test_bc)

    joblib.dump(knn_bc, os.path.join(config.MODELS_DIR, "exp07_knn_breastcancer.joblib"))

    fpr_d, tpr_d, _ = roc_curve(y_test_d, y_proba_d)
    metrics["exp07"] = {
        "diabetes": {
            "best_k": best_k,
            "k_values": k_range,
            "cv_scores": [round(s, 4) for s in cv_scores],
            "accuracy": float(accuracy_score(y_test_d, y_pred_d)),
            "precision": float(precision_score(y_test_d, y_pred_d)),
            "recall": float(recall_score(y_test_d, y_pred_d)),
            "f1": float(f1_score(y_test_d, y_pred_d)),
            "auc": float(roc_auc_score(y_test_d, y_proba_d)),
            "confusion_matrix": confusion_matrix(y_test_d, y_pred_d).tolist(),
            "classes": ["No Diabetes (0)", "Diabetes (1)"],
            "features": list(X_diab.columns),
            "roc_fpr": [round(float(x), 3) for x in fpr_d[::max(1, len(fpr_d)//15)]],
            "roc_tpr": [round(float(y), 3) for y in tpr_d[::max(1, len(tpr_d)//15)]]
        },
        "breast_cancer": {
            "accuracy": float(accuracy_score(y_test_bc, y_pred_bc)),
            "features": mean_cols,
            "classes": ["Benign (0)", "Malignant (1)"]
        }
    }
    print("✓ Exp 07 trained and saved")

    # -------------------------------------------------------------
    # EXP 08: Gaussian Naive Bayes (Banknote Authentication)
    # -------------------------------------------------------------
    bank_path = os.path.join(config.DATA_RAW_DIR, "data_banknote_authentication.csv")
    df_bank = pd.read_csv(bank_path)
    X_bank = df_bank.iloc[:, :4]
    y_bank = df_bank.iloc[:, 4]
    feature_names_bank = ["variance", "skewness", "curtosis", "entropy"]
    X_bank.columns = feature_names_bank
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank)

    gnb = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ])
    gnb.fit(X_train_b, y_train_b)
    y_pred_b = gnb.predict(X_test_b)
    y_proba_b = gnb.predict_proba(X_test_b)[:, 1]

    joblib.dump(gnb, os.path.join(config.MODELS_DIR, "exp08_gaussian_nb.joblib"))

    # Compute Gaussian parameters (mean, var) for visualization
    class_0_stats = {
        col: {"mean": float(X_train_b[y_train_b == 0][col].mean()), "std": float(X_train_b[y_train_b == 0][col].std())}
        for col in feature_names_bank
    }
    class_1_stats = {
        col: {"mean": float(X_train_b[y_train_b == 1][col].mean()), "std": float(X_train_b[y_train_b == 1][col].std())}
        for col in feature_names_bank
    }

    metrics["exp08"] = {
        "accuracy": float(accuracy_score(y_test_b, y_pred_b)),
        "precision": float(precision_score(y_test_b, y_pred_b)),
        "recall": float(recall_score(y_test_b, y_pred_b)),
        "f1": float(f1_score(y_test_b, y_pred_b)),
        "confusion_matrix": confusion_matrix(y_test_b, y_pred_b).tolist(),
        "classes": ["Counterfeit (0)", "Genuine (1)"],
        "features": feature_names_bank,
        "gaussian_stats_class0": class_0_stats,
        "gaussian_stats_class1": class_1_stats
    }
    print("✓ Exp 08 trained and saved")

    # -------------------------------------------------------------
    # EXP 09: Multinomial Naive Bayes (Campus Message Detective)
    # -------------------------------------------------------------
    # 1. Authentic 56-message Teaching Dataset
    msg_path = os.path.join(config.DATA_RAW_DIR, "campus_56_messages.csv")
    df_msg = pd.read_csv(msg_path)
    X_msg = df_msg["message"]
    y_msg = df_msg["label"]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_msg, y_msg, test_size=0.25, random_state=42, stratify=y_msg
    )
    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    X_train_bow = vectorizer.fit_transform(X_train_m)
    X_test_bow = vectorizer.transform(X_test_m)

    mnb = MultinomialNB()
    mnb.fit(X_train_bow, y_train_m)
    y_pred_mnb = mnb.predict(X_test_bow)

    # Save vectorizer and model bundle
    bundle_mnb = {
        "vectorizer": vectorizer,
        "model": mnb,
        "classes": list(mnb.classes_)
    }
    joblib.dump(bundle_mnb, os.path.join(config.MODELS_DIR, "exp09_multinomial_nb.joblib"))

    # Top words associated with each class
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    for class_idx, class_name in enumerate(mnb.classes_):
        log_probs = mnb.feature_log_prob_[class_idx]
        top_idx = np.argsort(log_probs)[-10:][::-1]
        top_words[str(class_name)] = [
            {"word": feature_names[i], "log_prob": round(float(log_probs[i]), 3)}
            for i in top_idx
        ]

    # Also train 222-message expanded dataset as switcher
    msg222_path = os.path.join(config.DATA_RAW_DIR, "college_messages_222.csv")
    df_222 = pd.read_csv(msg222_path)
    vec222 = CountVectorizer(lowercase=True, stop_words="english")
    X_train_222, X_test_222, y_train_222, y_test_222 = train_test_split(
        df_222["message"], df_222["label"], test_size=0.25, random_state=42, stratify=df_222["label"]
    )
    X_train_bow222 = vec222.fit_transform(X_train_222)
    X_test_bow222 = vec222.transform(X_test_222)
    mnb222 = MultinomialNB()
    mnb222.fit(X_train_bow222, y_train_222)
    y_pred_222 = mnb222.predict(X_test_bow222)
    bundle_222 = {
        "vectorizer": vec222,
        "model": mnb222,
        "classes": list(mnb222.classes_)
    }
    joblib.dump(bundle_222, os.path.join(config.MODELS_DIR, "exp09_multinomial_nb_222.joblib"))

    metrics["exp09"] = {
        "accuracy": float(accuracy_score(y_test_m, y_pred_mnb)),
        "precision": float(precision_score(y_test_m, y_pred_mnb, pos_label="Important", zero_division=0)),
        "recall": float(recall_score(y_test_m, y_pred_mnb, pos_label="Important", zero_division=0)),
        "f1": float(f1_score(y_test_m, y_pred_mnb, pos_label="Important", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_m, y_pred_mnb, labels=["Important", "Promotional"]).tolist(),
        "classes": ["Important", "Promotional"],
        "top_words": top_words,
        "vocab_size": len(feature_names),
        "total_messages": len(df_msg),
        "extended_dataset": {
            "accuracy": float(accuracy_score(y_test_222, y_pred_222)),
            "classes": list(mnb222.classes_),
            "macro_f1": float(f1_score(y_test_222, y_pred_222, average="macro", zero_division=0))
        }
    }
    print("[OK] Exp 09 trained and saved")

    # -------------------------------------------------------------
    # EXP 10: Linear SVM (Banknote Authentication)
    # -------------------------------------------------------------
    svm_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='linear', probability=True, random_state=42))
    ])
    svm_linear.fit(X_train_b, y_train_b)
    y_pred_svm_l = svm_linear.predict(X_test_b)
    y_proba_svm_l = svm_linear.predict_proba(X_test_b)[:, 1]

    joblib.dump(svm_linear, os.path.join(config.MODELS_DIR, "exp10_linear_svm.joblib"))

    # Compute 2D Decision Boundary projection for Linear SVM
    c_mean = float(df_bank['curtosis'].mean())
    e_mean = float(df_bank['entropy'].mean())
    boundary_line = []
    margin_plus = []
    margin_minus = []
    from scipy.optimize import brentq
    for x_val in np.linspace(-4.5, 4.5, 30):
        def dec_func_l(y, target=0):
            df_in = pd.DataFrame([[x_val, y, c_mean, e_mean]], columns=feature_names_bank)
            return float(svm_linear.decision_function(df_in)[0]) - target

        try:
            y_0 = brentq(dec_func_l, -15, 15, args=(0,))
            boundary_line.append({"x": round(float(x_val), 2), "y": round(float(y_0), 2)})
        except Exception:
            pass
        try:
            y_p = brentq(dec_func_l, -15, 15, args=(1,))
            margin_plus.append({"x": round(float(x_val), 2), "y": round(float(y_p), 2)})
        except Exception:
            pass
        try:
            y_m = brentq(dec_func_l, -15, 15, args=(-1,))
            margin_minus.append({"x": round(float(x_val), 2), "y": round(float(y_m), 2)})
        except Exception:
            pass

    c0_sub = df_bank[df_bank['class'] == 0].sample(n=35, random_state=42)
    c1_sub = df_bank[df_bank['class'] == 1].sample(n=35, random_state=42)
    class0_pts = [{"x": round(float(r['variance']), 2), "y": round(float(r['skewness']), 2)} for _, r in c0_sub.iterrows()]
    class1_pts = [{"x": round(float(r['variance']), 2), "y": round(float(r['skewness']), 2)} for _, r in c1_sub.iterrows()]

    scaler_l = svm_linear.named_steps['scaler']
    sv_raw_l = scaler_l.inverse_transform(svm_linear.named_steps['model'].support_vectors_)
    sv_pts_l = [{"x": round(float(v[0]), 2), "y": round(float(v[1]), 2)} for v in sv_raw_l[:18]]

    metrics["exp10"] = {
        "accuracy": float(accuracy_score(y_test_b, y_pred_svm_l)),
        "precision": float(precision_score(y_test_b, y_pred_svm_l)),
        "recall": float(recall_score(y_test_b, y_pred_svm_l)),
        "f1": float(f1_score(y_test_b, y_pred_svm_l)),
        "confusion_matrix": confusion_matrix(y_test_b, y_pred_svm_l).tolist(),
        "classes": ["Counterfeit (0)", "Genuine (1)"],
        "features": feature_names_bank,
        "support_vectors_count": int(len(svm_linear.named_steps['model'].support_vectors_)),
        "decision_boundary": {
            "boundary_line": boundary_line,
            "margin_plus": margin_plus,
            "margin_minus": margin_minus,
            "class0_samples": class0_pts,
            "class1_samples": class1_pts,
            "support_vectors": sv_pts_l,
            "x_label": "Wavelet Variance (X1)",
            "y_label": "Wavelet Skewness (X2)"
        }
    }
    print("✓ Exp 10 trained and saved")

    # -------------------------------------------------------------
    # EXP 11: Non-Linear SVM (RBF Kernel - Banknote Authentication)
    # -------------------------------------------------------------
    svm_rbf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])
    svm_rbf.fit(X_train_b, y_train_b)
    y_pred_svm_r = svm_rbf.predict(X_test_b)
    y_proba_svm_r = svm_rbf.predict_proba(X_test_b)[:, 1]

    joblib.dump(svm_rbf, os.path.join(config.MODELS_DIR, "exp11_nonlinear_svm.joblib"))

    # Compute RBF Decision Boundary curve
    boundary_curve = []
    for x_val in np.linspace(-4.5, 4.5, 35):
        def dec_func_rbf(y):
            df_in = pd.DataFrame([[x_val, y, c_mean, e_mean]], columns=feature_names_bank)
            return float(svm_rbf.decision_function(df_in)[0])

        try:
            y_root = brentq(dec_func_rbf, -15, 15)
            boundary_curve.append({"x": round(float(x_val), 2), "y": round(float(y_root), 2)})
        except Exception:
            pass

    scaler_r = svm_rbf.named_steps['scaler']
    sv_raw_r = scaler_r.inverse_transform(svm_rbf.named_steps['model'].support_vectors_)
    sv_pts_r = [{"x": round(float(v[0]), 2), "y": round(float(v[1]), 2)} for v in sv_raw_r[:18]]

    metrics["exp11"] = {
        "accuracy": float(accuracy_score(y_test_b, y_pred_svm_r)),
        "precision": float(precision_score(y_test_b, y_pred_svm_r)),
        "recall": float(recall_score(y_test_b, y_pred_svm_r)),
        "f1": float(f1_score(y_test_b, y_pred_svm_r)),
        "confusion_matrix": confusion_matrix(y_test_b, y_pred_svm_r).tolist(),
        "classes": ["Counterfeit (0)", "Genuine (1)"],
        "features": feature_names_bank,
        "support_vectors_count": int(len(svm_rbf.named_steps['model'].support_vectors_)),
        "comparison_with_linear": {
            "linear_accuracy": float(accuracy_score(y_test_b, y_pred_svm_l)),
            "rbf_accuracy": float(accuracy_score(y_test_b, y_pred_svm_r)),
            "linear_sv_count": int(len(svm_linear.named_steps['model'].support_vectors_)),
            "rbf_sv_count": int(len(svm_rbf.named_steps['model'].support_vectors_))
        },
        "decision_boundary": {
            "boundary_curve": boundary_curve,
            "class0_samples": class0_pts,
            "class1_samples": class1_pts,
            "support_vectors": sv_pts_r,
            "x_label": "Wavelet Variance (X1)",
            "y_label": "Wavelet Skewness (X2)"
        }
    }
    print("✓ Exp 11 trained and saved")

    # -------------------------------------------------------------
    # EXP 12: Random Forest Classification (Diabetes & Heart Disease)
    # -------------------------------------------------------------
    rf_diab = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf_diab.fit(X_train_d, y_train_d)
    y_pred_rf_d = rf_diab.predict(X_test_d)
    y_proba_rf_d = rf_diab.predict_proba(X_test_d)[:, 1]

    joblib.dump(rf_diab, os.path.join(config.MODELS_DIR, "exp12_rf_classifier.joblib"))

    rf_importances = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(X_diab.columns, rf_diab.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )
    # Compare with single Decision Tree on the exact same Diabetes split
    dt_diab = DecisionTreeClassifier(random_state=42)
    dt_diab.fit(X_train_d, y_train_d)
    y_pred_dt_d = dt_diab.predict(X_test_d)

    metrics["exp12"] = {
        "accuracy": float(accuracy_score(y_test_d, y_pred_rf_d)),
        "precision": float(precision_score(y_test_d, y_pred_rf_d)),
        "recall": float(recall_score(y_test_d, y_pred_rf_d)),
        "f1": float(f1_score(y_test_d, y_pred_rf_d)),
        "auc": float(roc_auc_score(y_test_d, y_proba_rf_d)),
        "oob_score": float(rf_diab.oob_score_),
        "confusion_matrix": confusion_matrix(y_test_d, y_pred_rf_d).tolist(),
        "classes": ["No Diabetes (0)", "Diabetes (1)"],
        "feature_importances": rf_importances,
        "features": list(X_diab.columns),
        "comparison_decision_tree": {
            "dt_accuracy": float(accuracy_score(y_test_d, y_pred_dt_d)),
            "rf_accuracy": float(accuracy_score(y_test_d, y_pred_rf_d)),
            "dt_f1": float(f1_score(y_test_d, y_pred_dt_d)),
            "rf_f1": float(f1_score(y_test_d, y_pred_rf_d))
        }
    }
    print("✓ Exp 12 trained and saved")

    # -------------------------------------------------------------
    # EXP 13: Random Forest Regression (Glucose Prediction)
    # -------------------------------------------------------------
    # Predict Glucose from other physical & medical measurements
    X_reg = df_diab_clean.drop(columns=["Glucose", "Outcome"])
    y_reg = df_diab_clean["Glucose"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)
    y_pred_r = rf_reg.predict(X_test_r)

    joblib.dump(rf_reg, os.path.join(config.MODELS_DIR, "exp13_rf_regressor.joblib"))

    residuals = [round(float(a - p), 2) for a, p in zip(y_test_r.iloc[:20], y_pred_r[:20])]
    rf_reg_importances = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(X_reg.columns, rf_reg.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )

    metrics["exp13"] = {
        "mae": float(mean_absolute_error(y_test_r, y_pred_r)),
        "mse": float(mean_squared_error(y_test_r, y_pred_r)),
        "r2": float(r2_score(y_test_r, y_pred_r)),
        "target": "Glucose (mg/dL)",
        "features": list(X_reg.columns),
        "feature_importances": rf_reg_importances,
        "actual": [round(float(v), 2) for v in y_test_r.iloc[:20].tolist()],
        "predicted": [round(float(v), 2) for v in y_pred_r[:20].tolist()],
        "residuals": residuals
    }
    print("✓ Exp 13 trained and saved")

    # Save metrics store
    with open(config.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("--- All 13 Experiment Models & Metrics Successfully Generated! ---")

if __name__ == "__main__":
    train_all()
