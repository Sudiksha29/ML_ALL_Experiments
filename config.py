import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "generated", "reports")
METRICS_FILE = os.path.join(BASE_DIR, "training", "metrics_store.json")
HISTORY_DB = os.path.join(BASE_DIR, "data", "experiment_history.sqlite")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "training"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

STUDENT_INFO = {
    "name": "Sudiksha Mahesh Ingale",
    "college": "D. Y. Patil College of Engineering and Technology, Kolhapur",
    "department": "Computer Science and Engineering — Artificial Intelligence and Machine Learning",
    "roll_number": "03",
    "batch": "TA1",
    "email": "sudikshaingale@gmail.com",
    "mobile": "9307083803",
    "guide": "Dr. Sachin Takmare",
    "subject": "Machine Learning Laboratory",
    "project_name": "ML LabVerse",
    "tagline": "Your Interactive Machine Learning Studio",
    "academic_year": "2025–2026",
    "semester": "Semester VI",
    "github": "https://github.com/Sudiksha29",
    "github_repo": "https://github.com/Sudiksha29/ML_ALL_Experiments",
    "linkedin": "https://www.linkedin.com/in/sudiksha-ingale-3bb37232b",
    "profile_image": "/static/images/profile.jpg",
    "college_logo": "/static/images/college_logo.png"
}

EXPERIMENTS_META = {
    "exp01": {
        "id": "exp01",
        "num": "01",
        "title": "Dataset Exploration & Visualization",
        "short_name": "Dataset Exploration",
        "category": "Data Engineering",
        "category_badge": "Exploration",
        "type": "exploratory",
        "algorithm": "Pandas, NumPy, Matplotlib & Seaborn EDA",
        "description": "Comprehensive exploratory data analysis pipeline covering schema inspection, null value profiling, statistical summaries, distribution histograms, and multi-variable correlation heatmaps.",
        "icon": "bar-chart-2",
        "color_accent": "teal",
        "dataset_name": "Multi-Dataset Suite (Placement, Diabetes, Titanic, Heart, Banknote)",
        "datasets_available": [
            {"id": "placement", "name": "Student Placement Dataset", "file": "Placement.csv"},
            {"id": "diabetes", "name": "Pima Indians Diabetes Dataset", "file": "diabetes.csv"},
            {"id": "heart", "name": "UCI Heart Disease Cleveland Dataset", "file": "heart.csv"},
            {"id": "titanic", "name": "Titanic Passenger Survival Dataset", "file": "titanic.csv"},
            {"id": "banknote", "name": "Banknote Authentication Dataset", "file": "data_banknote_authentication.csv"},
            {"id": "realestate", "name": "Real Estate Valuation Dataset", "file": "Real estate - Real estate.csv"}
        ]
    },
    "exp02": {
        "id": "exp02",
        "num": "02",
        "title": "Simple Linear Regression",
        "short_name": "Simple Linear Regression",
        "category": "Regression",
        "category_badge": "Regression",
        "type": "regression",
        "algorithm": "LinearRegression (Ordinary Least Squares)",
        "description": "Models the linear relationship between a single predictor and a continuous numerical target via Ordinary Least Squares, fitting the canonical line equation y = mx + c.",
        "icon": "trending-up",
        "color_accent": "teal",
        "dataset_name": "Placement Package (LPA) vs CGPA & Study Hours vs Exam Score",
        "switcher_options": [
            {"id": "placement", "label": "CGPA → Placement Package (LPA)", "dataset": "Placement.csv"},
            {"id": "hours", "label": "Study Hours → Exam Score", "dataset": "Study Hours Notebook"}
        ]
    },
    "exp03": {
        "id": "exp03",
        "num": "03",
        "title": "Multiple Linear Regression",
        "short_name": "Multiple Linear Regression",
        "category": "Regression",
        "category_badge": "Regression",
        "type": "regression",
        "algorithm": "LinearRegression (Multivariate Normal Equation)",
        "description": "Estimates continuous dependent variables from multiple collinear or independent numerical features, decomposing partial regression slopes and multidimensional intercepts.",
        "icon": "git-branch",
        "color_accent": "blue",
        "dataset_name": "CGPA + IQ → Package (LPA) & Real Estate Valuation",
        "switcher_options": [
            {"id": "cgpa_iq", "label": "CGPA & IQ → Placement Package", "dataset": "Cgpa_iq_package.csv"},
            {"id": "realestate", "label": "Property Features → Unit Price", "dataset": "Real estate - Real estate.csv"}
        ]
    },
    "exp04": {
        "id": "exp04",
        "num": "04",
        "title": "Binary Logistic Regression",
        "short_name": "Binary Logistic Regression",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "LogisticRegression (Sigmoid Activation)",
        "description": "Calculates probabilistic class memberships for binary target variables using the log-odds logit link function and Maximum Likelihood Estimation.",
        "icon": "check-circle",
        "color_accent": "peach",
        "dataset_name": "Titanic Passenger Survival & Student Placement Binary",
        "switcher_options": [
            {"id": "titanic", "label": "Titanic Passenger Survival (Pclass, Sex, Age, Fare)", "dataset": "titanic.csv"},
            {"id": "placement", "label": "Placement Status (CGPA, IQ)", "dataset": "Placement_Y_N.csv"}
        ]
    },
    "exp05": {
        "id": "exp05",
        "num": "05",
        "title": "Multiclass Logistic Regression",
        "short_name": "Multiclass Logistic Regression",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "LogisticRegression (Softmax / Multinomial Cross-Entropy)",
        "description": "Generalizes binary logistic regression across three or more mutually exclusive target categories utilizing normalized exponential softmax distributions.",
        "icon": "layers",
        "color_accent": "lavender",
        "dataset_name": "Student Performance Level & Iris Flower Species",
        "switcher_options": [
            {"id": "student", "label": "Study Hours & Attendance → Performance (Low/Medium/High)", "dataset": "Student Performance Practical"},
            {"id": "iris", "label": "Morphological Measurements → Iris Species (Setosa/Versicolor/Virginica)", "dataset": "Iris Dataset"}
        ]
    },
    "exp06": {
        "id": "exp06",
        "num": "06",
        "title": "Decision Tree Classification",
        "short_name": "Decision Tree",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "DecisionTreeClassifier (Gini Impurity Criterion)",
        "description": "Recursive partitioning algorithm generating interpretable hierarchical decision boundaries via Gini index impurity reduction.",
        "icon": "git-merge",
        "color_accent": "teal",
        "dataset_name": "UCI Heart Disease Cleveland Dataset (13 Clinical Predictors)",
        "disclaimer": "Educational Demonstration Only: This machine learning model is intended strictly for academic experimentation and must not be used as a clinical diagnostic tool."
    },
    "exp07": {
        "id": "exp07",
        "num": "07",
        "title": "K-Nearest Neighbors (KNN)",
        "short_name": "KNN Classification",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "KNeighborsClassifier (Minkowski / Euclidean Metric)",
        "description": "Instance-based non-parametric learning algorithm that infers target class through majority voting among the K closest standardized training samples.",
        "icon": "users",
        "color_accent": "blue",
        "dataset_name": "Pima Indians Diabetes & Breast Cancer Diagnostic",
        "switcher_options": [
            {"id": "diabetes", "label": "Pima Indians Diabetes (8 Clinical Biomarkers, K=17)", "dataset": "diabetes.csv"},
            {"id": "breast_cancer", "label": "Wisconsin Breast Cancer (Mean Diagnostic Measurements)", "dataset": "breast-cancer.csv"}
        ],
        "disclaimer": "Educational Health Simulation: Not certified for diagnostic or medical decision-making."
    },
    "exp08": {
        "id": "exp08",
        "num": "08",
        "title": "Gaussian Naive Bayes",
        "short_name": "Gaussian Naive Bayes",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "GaussianNB (Normal Distribution Likelihood)",
        "description": "Probabilistic classifier founded upon Bayes' Theorem with the assumption of class-conditional feature independence following continuous Gaussian normal distributions.",
        "icon": "activity",
        "color_accent": "peach",
        "dataset_name": "Banknote Authentication Physical Measurements (Variance, Skewness, Curtosis, Entropy)"
    },
    "exp09": {
        "id": "exp09",
        "num": "09",
        "title": "Multinomial Naive Bayes",
        "short_name": "Multinomial Naive Bayes",
        "category": "NLP & Text",
        "category_badge": "NLP & Text",
        "type": "nlp",
        "algorithm": "MultinomialNB + CountVectorizer (Bag-of-Words)",
        "description": "Natural Language Processing classifier converting tokenized message corpora into Bag-of-Words frequency vectors and computing class log-likelihoods with Laplace smoothing.",
        "icon": "message-square",
        "color_accent": "lavender",
        "dataset_name": "Campus Message Detective Teaching Dataset (56 College Messages)",
        "disclaimer": "Academic Teaching Dataset: Designed specifically to teach discrete word-frequency likelihoods rather than commercial spam filtering."
    },
    "exp10": {
        "id": "exp10",
        "num": "10",
        "title": "Linear Support Vector Machine",
        "short_name": "Linear SVM",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "SVC(kernel='linear')",
        "description": "Finds the optimal separating hyperplane that maximizes the geometric margin between genuine and forged class support vectors in normalized Euclidean space.",
        "icon": "shield-check",
        "color_accent": "teal",
        "dataset_name": "Banknote Authentication Dataset (Linear Hyperplane)"
    },
    "exp11": {
        "id": "exp11",
        "num": "11",
        "title": "Non-Linear Support Vector Machine",
        "short_name": "Non-Linear SVM (RBF)",
        "category": "Classification",
        "category_badge": "Classification",
        "type": "classification",
        "algorithm": "SVC(kernel='rbf')",
        "description": "Applies the Radial Basis Function (RBF) kernel trick to map non-separable continuous feature spaces into infinite-dimensional Hilbert space for non-linear boundary separation.",
        "icon": "compass",
        "color_accent": "blue",
        "dataset_name": "Banknote Authentication (RBF Kernel Mapping)"
    },
    "exp12": {
        "id": "exp12",
        "num": "12",
        "title": "Random Forest Classification",
        "short_name": "Random Forest Classifier",
        "category": "Classification",
        "category_badge": "Ensemble",
        "type": "classification",
        "algorithm": "RandomForestClassifier (Ensemble Bagging)",
        "description": "Meta-estimator combining 100 decorrelated decision trees trained on bootstrap data subsets to minimize variance and combat individual tree overfitting.",
        "icon": "share-2",
        "color_accent": "lavender",
        "dataset_name": "Pima Indians Diabetes Diagnostic (Ensemble of 100 Trees)"
    },
    "exp13": {
        "id": "exp13",
        "num": "13",
        "title": "Random Forest Regression",
        "short_name": "Random Forest Regressor",
        "category": "Regression",
        "category_badge": "Ensemble",
        "type": "regression",
        "algorithm": "RandomForestRegressor (Ensemble Bagging)",
        "description": "Predicts continuous metabolic targets by aggregating numerical outputs across an ensemble of 100 randomized regression sub-trees.",
        "icon": "bar-chart",
        "color_accent": "peach",
        "dataset_name": "Blood Glucose Level Prediction (Metabolic Biomarkers → Glucose mg/dL)"
    }
}
