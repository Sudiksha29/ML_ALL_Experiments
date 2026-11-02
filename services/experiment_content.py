"""
ML LabVerse — Experiment Academic Content & Complete Viva Voce Guide
Contains:
1. Exact Algorithm Mathematical Formulations & Component Legends
2. Authentic Viva Questions & Answers matching external viva exam specifications
"""

EXPERIMENT_DEEP_DIVE = {
    "exp01": {
        "problem_statement": "Dataset Exploration and Exploratory Data Analysis (EDA) is the foundational phase of understanding data using statistics and visualizations before training any machine learning model.",
        "dataset_name": "Multi-Dataset Laboratory Repository (14 Tabular Datasets)",
        "features": [
            {"name": "Independent Attributes (X)", "type": "Continuous / Categorical", "unit": "Varies by dataset", "desc": "Raw observable measurements, physical biomarkers, socioeconomic indicators, or text.", "range": "Dataset specific"},
            {"name": "Target Variable (y)", "type": "Continuous / Discrete", "unit": "Varies by practical", "desc": "Ground-truth target outcome for supervised regression or classification practicals.", "range": "Continuous or {0, 1, ..., K-1}"}
        ],
        "formula": {
            "title": "Exploratory Data Analysis Statistical Formulas",
            "equation": "x̄ = (1/n) Σ xᵢ   |   s = √[ (1/(n-1)) Σ (xᵢ - x̄)² ]   |   r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[ Σ(xᵢ-x̄)² Σ(yᵢ-ȳ)² ]",
            "components": [
                {"symbol": "x̄", "meaning": "Sample Mean (arithmetic average of values)"},
                {"symbol": "s", "meaning": "Sample Standard Deviation (measure of dispersion)"},
                {"symbol": "r", "meaning": "Pearson Correlation Coefficient (-1.0 to +1.0)"},
                {"symbol": "n", "meaning": "Total number of observations (rows in dataset)"}
            ],
            "objective": "Understand data distributions, audit missing values, and check feature correlations."
        },
        "viva_qa": [
            {"q": "What is dataset exploration?", "a": "It is the process of understanding a dataset before training a machine learning model."},
            {"q": "What is EDA?", "a": "EDA stands for Exploratory Data Analysis. It helps us understand data using statistics and visualizations."},
            {"q": "What is Pandas?", "a": "Pandas is a Python library used for data manipulation and analysis."},
            {"q": "What is a DataFrame?", "a": "A DataFrame is a two-dimensional tabular data structure with rows and columns."},
            {"q": "What does df.head() do?", "a": "Displays the first five rows by default."},
            {"q": "What does df.tail() do?", "a": "Displays the last five rows by default."},
            {"q": "What does df.shape return?", "a": "It returns the number of rows and columns."},
            {"q": "What does df.info() do?", "a": "Displays column names, data types, and non-null counts."},
            {"q": "What does df.describe() do?", "a": "Displays statistical information such as mean, standard deviation, minimum, and maximum for numerical columns."},
            {"q": "How do you check missing values?", "a": "Using df.isnull().sum()"},
            {"q": "What is a histogram?", "a": "A graph that shows the frequency distribution of numerical data."},
            {"q": "What is a scatter plot?", "a": "A graph used to show the relationship between two numerical variables."},
            {"q": "What is a correlation matrix?", "a": "A table that shows the correlation between numerical variables."},
            {"q": "What is Seaborn?", "a": "Seaborn is a Python visualization library built on Matplotlib that makes statistical plots easier to create."}
        ],
        "conclusion": "Exploratory analysis confirmed the dataset schema, validated absence of missing values across clean sets, and established empirical feature distributions prior to modeling."
    },

    "exp02": {
        "problem_statement": "Simple Linear Regression is a supervised learning algorithm used to predict continuous numerical values using only one independent variable.",
        "dataset_name": "Placement.csv (200 Student Records) & Study Hours Practical",
        "features": [
            {"name": "cgpa", "type": "Continuous Float", "unit": "Grade Point (0-10)", "desc": "Cumulative Grade Point Average of undergraduate engineering student.", "range": "5.00 – 10.00"},
            {"name": "hours", "type": "Continuous Float", "unit": "Hours per day", "desc": "Average dedicated daily study hours.", "range": "1.0 – 15.0"},
            {"name": "package (Target)", "type": "Continuous Float", "unit": "LPA", "desc": "Annual salary package secured during campus recruitment.", "range": "1.37 – 4.62 LPA"}
        ],
        "formula": {
            "title": "Simple Linear Regression Equation",
            "equation": "y = mx + c",
            "components": [
                {"symbol": "y", "meaning": "Predicted output"},
                {"symbol": "x", "meaning": "Input feature"},
                {"symbol": "m", "meaning": "Slope (how much y changes when x increases by 1 unit)"},
                {"symbol": "c", "meaning": "Intercept (predicted output when x = 0)"}
            ],
            "objective": "Find the best-fitting line that minimizes the sum of squared residuals (OLS)."
        },
        "viva_qa": [
            {"q": "What is Linear Regression?", "a": "Linear Regression is a supervised learning algorithm used to predict continuous numerical values."},
            {"q": "What is Simple Linear Regression?", "a": "It predicts a numerical output using only one independent variable."},
            {"q": "What is its equation?", "a": "y = mx + c (y: Predicted output, x: Input feature, m: Slope, c: Intercept)."},
            {"q": "What is the slope?", "a": "Slope represents how much the predicted output changes when the input increases by one unit."},
            {"q": "What is the intercept?", "a": "It is the predicted output when the input value is zero."},
            {"q": "Which Python class is used?", "a": "from sklearn.linear_model import LinearRegression"},
            {"q": "What is the main objective of Linear Regression?", "a": "To find the best-fitting line that minimizes the sum of squared residuals."},
            {"q": "What is a residual?", "a": "The difference between the actual value and predicted value."},
            {"q": "What is MSE?", "a": "Mean Squared Error measures the average of squared prediction errors. Lower values generally indicate better predictions on the same target scale."},
            {"q": "What is R² score?", "a": "R² measures how much of the variation in the target is explained by the model. A value closer to 1 generally indicates a better fit."},
            {"q": "Give an example of SLR.", "a": "Predicting a student's package using only their CGPA."},
            {"q": "What is the limitation of SLR?", "a": "It uses only one input feature and cannot capture complex nonlinear relationships using its basic form."}
        ],
        "conclusion": "Ordinary Least Squares derived a robust best-fit line. CGPA demonstrated a strong linear correlation with campus placement packages (R² > 0.82)."
    },

    "exp03": {
        "problem_statement": "Multiple Linear Regression predicts a continuous numerical output using two or more independent variables.",
        "dataset_name": "Cgpa_iq_package.csv (200 Records) & Real Estate Valuation",
        "features": [
            {"name": "cgpa", "type": "Continuous Float", "unit": "Grade Point (0-10)", "desc": "Academic performance indicator.", "range": "5.0 – 10.0"},
            {"name": "iq", "type": "Integer", "unit": "IQ Score Points", "desc": "Standardized cognitive intelligence score.", "range": "70 – 160"},
            {"name": "package (Target)", "type": "Continuous Float", "unit": "LPA", "desc": "Placement package continuous target.", "range": "1.37 – 4.62 LPA"}
        ],
        "formula": {
            "title": "Multiple Linear Regression Equation",
            "equation": "y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ",
            "components": [
                {"symbol": "y", "meaning": "Predicted continuous output"},
                {"symbol": "b₀", "meaning": "Y-Intercept (bias term)"},
                {"symbol": "b₁, b₂, ..., bₙ", "meaning": "Regression coefficients for each feature"},
                {"symbol": "x₁, x₂, ..., xₙ", "meaning": "Input independent features"}
            ],
            "objective": "Decompose partial contributions of multiple features toward a single continuous target."
        },
        "viva_qa": [
            {"q": "What is Multiple Linear Regression?", "a": "It predicts a continuous numerical output using two or more independent variables."},
            {"q": "Difference between SLR and MLR?", "a": "SLR uses one input feature. MLR uses multiple input features."},
            {"q": "What is its equation?", "a": "y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ"},
            {"q": "Give an example.", "a": "Predicting a student's package using CGPA, IQ, and technical skills."},
            {"q": "Which algorithm is used?", "a": "from sklearn.linear_model import LinearRegression"},
            {"q": "Can LinearRegression handle multiple features?", "a": "Yes. The same LinearRegression() class is used for both SLR and MLR."},
            {"q": "What is multicollinearity?", "a": "It occurs when two or more input features are highly correlated with each other."},
            {"q": "Why can multicollinearity be a problem?", "a": "It can make individual regression coefficients unstable and difficult to interpret."},
            {"q": "Which metrics are used to evaluate MLR?", "a": "MSE and R² score are commonly used."},
            {"q": "Does MLR always give better results than SLR?", "a": "No. Adding more features does not guarantee better performance on unseen data."}
        ],
        "conclusion": "Incorporating both CGPA and IQ captured broader predictive signals, yielding an accurate multivariate continuous forecast."
    },

    "exp04": {
        "problem_statement": "Binary Logistic Regression is a supervised learning classification algorithm used to predict binary categories (such as Pass/Fail or Survived/Not Survived) using the sigmoid activation function.",
        "dataset_name": "titanic.csv (891 Passengers) & Placement_Y_N.csv",
        "features": [
            {"name": "pclass", "type": "Categorical Ordinal", "unit": "Ticket Class (1, 2, 3)", "desc": "Passenger cabin class.", "range": "1, 2, 3"},
            {"name": "sex", "type": "Binary Integer", "unit": "Gender Encoding", "desc": "1 = Female, 0 = Male.", "range": "0, 1"},
            {"name": "age", "type": "Continuous Float", "unit": "Years", "desc": "Passenger age.", "range": "0.42 – 80.0"},
            {"name": "fare", "type": "Continuous Float", "unit": "USD $", "desc": "Ticket fare.", "range": "0.0 – 512.33"},
            {"name": "survived (Target)", "type": "Binary Integer", "unit": "Class Label", "desc": "0 = Did not survive, 1 = Survived.", "range": "0, 1"}
        ],
        "formula": {
            "title": "Sigmoid (Logistic) Function",
            "equation": "σ(z) = 1 / (1 + e⁻ᶻ)    [where z = w·x + b]",
            "components": [
                {"symbol": "σ(z)", "meaning": "Predicted probability of positive class (between 0 and 1)"},
                {"symbol": "z", "meaning": "Linear combination of inputs (w·x + b)"},
                {"symbol": "w", "meaning": "Model weights / coefficients vector"},
                {"symbol": "b", "meaning": "Model bias / intercept term"}
            ],
            "objective": "Map unbounded real values z into calibrated probabilities in (0, 1) for binary decision-making."
        },
        "viva_qa": [
            {"q": "What is Logistic Regression?", "a": "It is a supervised learning algorithm used for classification."},
            {"q": "Why is it called Logistic Regression if it performs classification?", "a": "Because it uses a logistic function to estimate probabilities, which are then used to classify observations."},
            {"q": "What is binary classification?", "a": "Classification involving exactly two classes, such as Yes/No or Pass/Fail."},
            {"q": "What is the sigmoid function?", "a": "It converts a numerical value into a probability between 0 and 1: σ(z) = 1 / (1 + e⁻ᶻ)."},
            {"q": "What is the usual classification threshold?", "a": "0.5 is a common default threshold. Probability greater than or equal to 0.5 → Class 1. Probability below 0.5 → Class 0."},
            {"q": "Which class is used?", "a": "from sklearn.linear_model import LogisticRegression"},
            {"q": "What is predict_proba()?", "a": "It returns the predicted probability for each class."},
            {"q": "What is the difference between predict() and predict_proba()?", "a": "predict() returns the predicted class. predict_proba() returns class probabilities."},
            {"q": "What is a decision boundary?", "a": "A boundary that separates different predicted classes."},
            {"q": "Give an example.", "a": "Predicting whether a student will be placed or not placed."},
            {"q": "Is Logistic Regression a regression algorithm in practical usage?", "a": "Despite its name, it is primarily used for classification."}
        ],
        "conclusion": "The Sigmoid activation reliably estimated passenger survival probabilities on the Titanic dataset, achieving ~80% accuracy with an AUC of 0.86."
    },

    "exp05": {
        "problem_statement": "Multiclass Logistic Regression predicts one category from three or more possible mutually exclusive classes (such as Low, Medium, or High) using the Softmax normalization function.",
        "dataset_name": "Student Performance Practical & Iris Flower Dataset",
        "features": [
            {"name": "study_hours", "type": "Continuous Float", "unit": "Hours/day", "desc": "Dedicated preparation hours.", "range": "0 – 16 hrs"},
            {"name": "attendance", "type": "Continuous Float", "unit": "Percentage (%)", "desc": "Attendance percentage.", "range": "0 – 100%"},
            {"name": "performance (Target)", "type": "Multiclass Ordinal", "unit": "Class Label", "desc": "0 = Low, 1 = Medium, 2 = High.", "range": "0, 1, 2"}
        ],
        "formula": {
            "title": "Softmax Activation Function",
            "equation": "Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)",
            "components": [
                {"symbol": "zᵢ", "meaning": "Score (logit) for class i"},
                {"symbol": "Σⱼ e^(zⱼ)", "meaning": "Sum of exponential scores across all K classes"},
                {"symbol": "Softmax(zᵢ)", "meaning": "Predicted probability of class i (all probabilities sum to exactly 1.0)"}
            ],
            "objective": "Normalize multiple raw continuous class scores into a valid multi-category probability distribution."
        },
        "viva_qa": [
            {"q": "What is multiclass classification?", "a": "Predicting one category from three or more possible classes."},
            {"q": "Give an example.", "a": "Predicting student stress level as Low, Medium, or High."},
            {"q": "Difference between binary and multiclass Logistic Regression?", "a": "Binary predicts between two classes, while multiclass predicts among three or more classes."},
            {"q": "Does Scikit-learn use the same class for both?", "a": "Yes, both use LogisticRegression."},
            {"q": "What is the softmax function?", "a": "It converts multiple scores into probabilities that sum to 1."},
            {"q": "What is multi_class?", "a": "Historically, this parameter controlled the multiclass strategy in Logistic Regression. In current Scikit-learn versions, its use has changed and it is no longer a parameter to set in recent versions."},
            {"q": "What does predict_proba() return in multiclass classification?", "a": "A probability for each possible class."},
            {"q": "What is one-vs-rest (OvR)?", "a": "It trains one binary classifier for each class, separating that class from all remaining classes."},
            {"q": "Can Logistic Regression classify more than three classes?", "a": "Yes. It can handle multiple classes."}
        ],
        "conclusion": "The Softmax function cleanly separated the student performance tiers, outputting calibrated probability distributions summing to 1.0."
    },

    "exp06": {
        "problem_statement": "Decision Tree Classifier is a supervised learning algorithm that makes predictions using a hierarchical, tree-like structure of decisions, splitting nodes using the Gini Index impurity criterion.",
        "dataset_name": "UCI Heart Disease Cleveland Dataset (13 Clinical Features)",
        "features": [
            {"name": "age", "type": "Integer", "unit": "Years", "desc": "Patient age.", "range": "29 – 77"},
            {"name": "sex", "type": "Binary Integer", "unit": "Sex Indicator", "desc": "1 = Male, 0 = Female.", "range": "0, 1"},
            {"name": "cp", "type": "Categorical", "unit": "Pain Type (0-3)", "desc": "Chest pain type.", "range": "0 – 3"},
            {"name": "trestbps", "type": "Integer", "unit": "mm Hg", "desc": "Resting systolic blood pressure.", "range": "94 – 200"},
            {"name": "chol", "type": "Integer", "unit": "mg/dL", "desc": "Serum cholesterol.", "range": "126 – 564"},
            {"name": "fbs", "type": "Binary Integer", "unit": "Blood Sugar", "desc": "Fasting blood sugar > 120 mg/dl.", "range": "0, 1"},
            {"name": "restecg", "type": "Categorical", "unit": "ECG Code", "desc": "Resting ECG.", "range": "0 – 2"},
            {"name": "thalach", "type": "Integer", "unit": "BPM", "desc": "Max heart rate achieved.", "range": "71 – 202"},
            {"name": "exang", "type": "Binary Integer", "unit": "Angina Indicator", "desc": "Exercise induced angina.", "range": "0, 1"},
            {"name": "oldpeak", "type": "Continuous Float", "unit": "ST Depression", "desc": "ST depression induced by exercise.", "range": "0.0 – 6.2"},
            {"name": "slope", "type": "Categorical", "unit": "Slope Type", "desc": "Peak exercise ST slope.", "range": "0 – 2"},
            {"name": "ca", "type": "Integer", "unit": "Count", "desc": "Major vessels colored by fluoroscopy.", "range": "0 – 3"},
            {"name": "thal", "type": "Categorical", "unit": "Thal Code", "desc": "Thalassemia status.", "range": "1 – 3"}
        ],
        "formula": {
            "title": "Gini Impurity Index Formula",
            "equation": "Gini Index = 1 - Σ (pᵢ)²",
            "components": [
                {"symbol": "pᵢ", "meaning": "Proportion of samples belonging to class i at a given node"},
                {"symbol": "Gini = 0.0", "meaning": "Completely pure node (all samples belong to the exact same class)"},
                {"symbol": "Gini = 0.5", "meaning": "Equally distributed binary classes (maximum impurity)"}
            ],
            "objective": "Select the feature and threshold that maximizes impurity reduction at each internal split."
        },
        "viva_qa": [
            {"q": "What is a Decision Tree?", "a": "A supervised learning algorithm that makes predictions using a tree-like structure of decisions."},
            {"q": "How does it work?", "a": "It repeatedly splits the dataset based on feature conditions until it reaches a prediction at a leaf node."},
            {"q": "What are the main parts of a Decision Tree?", "a": "Root node – first decision; Decision/internal node – condition or test; Branch – outcome of a condition; Leaf node – final prediction."},
            {"q": "What is Gini Index?", "a": "Gini Index measures the impurity of a node. A lower value means the classes in that node are more pure."},
            {"q": "What is Entropy?", "a": "Entropy measures the uncertainty or impurity in a dataset."},
            {"q": "What is Information Gain?", "a": "It measures how much uncertainty is reduced after splitting the data."},
            {"q": "Which criterion is used in your Decision Tree practical?", "a": "Gini Index."},
            {"q": "Which class is used?", "a": "from sklearn.tree import DecisionTreeClassifier"},
            {"q": "What is max_depth?", "a": "It controls the maximum depth of the tree."},
            {"q": "What happens if the tree becomes too deep?", "a": "It may overfit the training data."},
            {"q": "What is pruning?", "a": "Pruning means reducing the size of a tree to help prevent overfitting."},
            {"q": "What are the advantages of Decision Trees?", "a": "Easy to understand, can model nonlinear relationships, requires little feature scaling."},
            {"q": "What is the main disadvantage?", "a": "A deep tree can overfit and small changes in training data can produce a different tree."}
        ],
        "conclusion": "The Decision Tree established interpretable hierarchical diagnostic rules on heart disease, identifying chest pain (cp) and major vessels (ca) as top split features."
    },

    "exp07": {
        "problem_statement": "K-Nearest Neighbors (KNN) is a supervised learning algorithm that predicts the class of a new observation using majority voting among the classes of its K nearest neighbors in feature space.",
        "dataset_name": "diabetes.csv (Pima Indians, K=17) & breast-cancer.csv",
        "features": [
            {"name": "Pregnancies", "type": "Integer", "unit": "Count", "desc": "Times pregnant.", "range": "0 – 17"},
            {"name": "Glucose", "type": "Integer", "unit": "mg/dL", "desc": "Plasma glucose concentration.", "range": "0 – 199"},
            {"name": "BloodPressure", "type": "Integer", "unit": "mm Hg", "desc": "Diastolic blood pressure.", "range": "0 – 122"},
            {"name": "SkinThickness", "type": "Integer", "unit": "mm", "desc": "Triceps skin fold thickness.", "range": "0 – 99"},
            {"name": "Insulin", "type": "Integer", "unit": "mu U/ml", "desc": "2-Hour serum insulin.", "range": "0 – 846"},
            {"name": "BMI", "type": "Continuous Float", "unit": "kg/m²", "desc": "Body mass index.", "range": "0.0 – 67.1"},
            {"name": "DiabetesPedigreeFunction", "type": "Continuous Float", "unit": "Score", "desc": "Genetic pedigree function.", "range": "0.078 – 2.42"},
            {"name": "Age", "type": "Integer", "unit": "Years", "desc": "Patient age.", "range": "21 – 81"}
        ],
        "formula": {
            "title": "Euclidean Distance Metric",
            "equation": "d = √[ Σᵢ₌₁ⁿ (xᵢ - yᵢ)² ]",
            "components": [
                {"symbol": "d", "meaning": "Straight-line geometric Euclidean distance between two data points"},
                {"symbol": "xᵢ", "meaning": "Feature value of the query observation in dimension i"},
                {"symbol": "yᵢ", "meaning": "Feature value of the stored training observation in dimension i"},
                {"symbol": "K", "meaning": "Number of nearest neighbors evaluated for majority voting"}
            ],
            "objective": "Identify the K nearest training instances using distance in standardized space."
        },
        "viva_qa": [
            {"q": "What is KNN?", "a": "KNN is a supervised learning algorithm that predicts the class of a new observation using the classes of its nearest neighbors."},
            {"q": "What does K represent?", "a": "K represents the number of nearest neighbors considered for prediction."},
            {"q": "How does KNN work?", "a": "1. Choose a value of K. 2. Calculate distances from the new point to training points. 3. Select the K nearest points. 4. Use majority voting to decide the predicted class."},
            {"q": "Which class is used?", "a": "from sklearn.neighbors import KNeighborsClassifier"},
            {"q": "What is Euclidean Distance?", "a": "It measures the straight-line distance between two points: d = √[ Σ(xᵢ - yᵢ)² ]."},
            {"q": "Why is KNN called a lazy learning algorithm?", "a": "Because it does not build a traditional model during training. It stores the training data and performs much of its work when a prediction is requested."},
            {"q": "Why is feature scaling important in KNN?", "a": "Because KNN uses distance. Features with larger numerical ranges can dominate the distance calculation."},
            {"q": "What is StandardScaler?", "a": "It standardizes features so they have approximately zero mean and unit standard deviation."},
            {"q": "What happens if K is too small?", "a": "The model can become sensitive to noise and overfit."},
            {"q": "What happens if K is too large?", "a": "The model may become too general and underfit."},
            {"q": "Why is an odd K often preferred for binary classification?", "a": "It can reduce the possibility of a tie between two classes."},
            {"q": "What is majority voting?", "a": "The class that occurs most frequently among the K nearest neighbors is selected."},
            {"q": "What is the disadvantage of KNN?", "a": "Prediction can be slow for large datasets because distances to many training observations must be calculated."}
        ],
        "conclusion": "Standardizing features was essential for KNN distance calculations. 10-fold cross-validation confirmed K=17 provides optimal balance on the Diabetes dataset."
    },

    "exp08": {
        "problem_statement": "Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem that assumes continuous numerical features follow a Gaussian (normal) distribution within each class.",
        "dataset_name": "data_banknote_authentication.csv (1,372 Authentic & Forged Notes)",
        "features": [
            {"name": "variance", "type": "Continuous Float", "unit": "Wavelet Variance", "desc": "Variance of Wavelet Transformed image.", "range": "-7.04 – 6.82"},
            {"name": "skewness", "type": "Continuous Float", "unit": "Wavelet Skewness", "desc": "Skewness of Wavelet Transformed image.", "range": "-13.77 – 12.95"},
            {"name": "curtosis", "type": "Continuous Float", "unit": "Wavelet Kurtosis", "desc": "Kurtosis of Wavelet Transformed image.", "range": "-5.28 – 17.92"},
            {"name": "entropy", "type": "Continuous Float", "unit": "Entropy", "desc": "Information entropy of image.", "range": "-8.58 – 2.45"}
        ],
        "formula": {
            "title": "Bayes' Theorem & Gaussian Normal Likelihood",
            "equation": "P(A|B) = [ P(B|A) · P(A) ] / P(B)   |   P(xᵢ|y) = [ 1 / √(2πσ²) ] · e^[ -(xᵢ - μ)² / (2σ²) ]",
            "components": [
                {"symbol": "P(A|B)", "meaning": "Posterior probability (probability of class after considering features)"},
                {"symbol": "P(B|A)", "meaning": "Likelihood (calculated using the normal Gaussian bell-curve equation)"},
                {"symbol": "P(A)", "meaning": "Prior probability of the class"},
                {"symbol": "μ, σ²", "meaning": "Mean and variance of continuous feature xᵢ in class y"}
            ],
            "objective": "Predict class label by maximizing posterior probability under conditional independence."
        },
        "viva_qa": [
            {"q": "What is Naive Bayes?", "a": "Naive Bayes is a supervised classification algorithm based on Bayes' theorem."},
            {"q": "Why is it called Naive?", "a": "Because it makes the simplifying assumption that features are conditionally independent given the class."},
            {"q": "State Bayes' theorem.", "a": "P(A|B) = [ P(B|A) · P(A) ] / P(B)"},
            {"q": "What is Gaussian Naive Bayes?", "a": "It is a Naive Bayes classifier that assumes numerical features follow a Gaussian (normal) distribution within each class."},
            {"q": "What is a Gaussian distribution?", "a": "It is a normal, bell-shaped probability distribution."},
            {"q": "Which class is used?", "a": "from sklearn.naive_bayes import GaussianNB"},
            {"q": "When do we use GaussianNB?", "a": "When the input features are numerical and the Gaussian distribution assumption is reasonable."},
            {"q": "What is prior probability?", "a": "The probability of a class before considering the input features."},
            {"q": "What is likelihood?", "a": "The probability of observing the given features under a particular class."},
            {"q": "What is posterior probability?", "a": "The probability of a class after considering the observed features."},
            {"q": "Why is Naive Bayes called a probabilistic algorithm?", "a": "Because it calculates class probabilities and predicts using those probabilities."},
            {"q": "What is its advantage?", "a": "It is simple, fast, and works well on many classification problems."},
            {"q": "What is its limitation?", "a": "Its independence assumption may not match real-world data where features are related."}
        ],
        "conclusion": "Gaussian Naive Bayes achieved fast, robust classification on currency authentication, successfully separating genuine from forged notes using bell-curve likelihoods."
    },

    "exp09": {
        "problem_statement": "Multinomial Naive Bayes is a classification algorithm commonly used for text classification based on word counts or similar non-negative features converted via CountVectorizer.",
        "dataset_name": "Campus Message Detective Teaching Dataset (56 College Messages)",
        "features": [
            {"name": "raw_text_message", "type": "String / Text", "unit": "English Text", "desc": "Message string containing academic alerts or promotional discounts.", "range": "Text"},
            {"name": "CountVectorizer Vocabulary", "type": "Sparse Word Vector", "unit": "Word Counts", "desc": "Bag-of-Words frequency vector.", "range": "Integer word counts"}
        ],
        "formula": {
            "title": "Laplace Smoothed Multinomial Probability",
            "equation": "P(wᵢ | c) = (N_ci + α) / (N_c + α·|V|)",
            "components": [
                {"symbol": "N_ci", "meaning": "Frequency of word wᵢ occurring in training messages of class c"},
                {"symbol": "N_c", "meaning": "Total count of all words in class c"},
                {"symbol": "|V|", "meaning": "Total vocabulary size (unique words learned by CountVectorizer)"},
                {"symbol": "α", "meaning": "Smoothing parameter (α = 1.0 corresponds to standard Laplace smoothing)"}
            ],
            "objective": "Prevent zero probabilities for unseen words in text classification."
        },
        "viva_qa": [
            {"q": "What is Multinomial Naive Bayes?", "a": "It is a Naive Bayes classification algorithm commonly used for text classification based on word counts or similar non-negative features."},
            {"q": "Where is it used?", "a": "Spam detection, sentiment analysis, news classification, message classification."},
            {"q": "Which class is used?", "a": "from sklearn.naive_bayes import MultinomialNB"},
            {"q": "What is CountVectorizer?", "a": "CountVectorizer converts text into numerical word-count features."},
            {"q": "Why do we need CountVectorizer?", "a": "Machine learning algorithms need numerical inputs. CountVectorizer converts text into numbers representing how many times words occur."},
            {"q": "What is a vocabulary?", "a": "It is the collection of unique words learned from the training text."},
            {"q": "Explain fit_transform().", "a": "It learns the vocabulary from the text and converts the text into a numerical count matrix."},
            {"q": "What is the difference between GaussianNB and MultinomialNB?", "a": "GaussianNB assumes Gaussian distribution and is used with continuous numerical features. MultinomialNB uses a multinomial feature model commonly used with word counts."},
            {"q": "What is Laplace smoothing?", "a": "It is a technique that prevents zero probabilities for words not observed in a particular class during training."},
            {"q": "What does alpha=1.0 mean?", "a": "It specifies the smoothing strength. alpha=1.0 corresponds to standard Laplace smoothing."},
            {"q": "Why can't MultinomialNB directly process raw text?", "a": "Because raw text must first be converted into suitable numerical features."},
            {"q": "Can MultinomialNB accept negative values?", "a": "No. Its standard input features must be non-negative."},
            {"q": "What is a Bag of Words model?", "a": "It represents text by counting words while ignoring grammar and usually word order."},
            {"q": "Explain this code: vectorizer = CountVectorizer(); X = vectorizer.fit_transform(messages)", "a": "It creates a vocabulary from the messages and converts each message into a word-count vector."},
            {"q": "Why do we save the vectorizer separately from the model?", "a": "Because new text must be transformed using the same learned vocabulary before it can be passed to the trained model."}
        ],
        "conclusion": "CountVectorizer paired with Multinomial Naive Bayes accurately discriminated campus notices from promotional offers using word-count likelihoods."
    },

    "exp10": {
        "problem_statement": "Linear Support Vector Machine is a supervised maximum-margin algorithm that constructs a linear decision boundary (hyperplane) to separate classes with the maximum possible margin.",
        "dataset_name": "data_banknote_authentication.csv (Linear Hyperplane)",
        "features": [
            {"name": "variance", "type": "Continuous Float", "unit": "Wavelet Variance", "desc": "Variance of Wavelet Transformed banknote image.", "range": "-7.04 – 6.82"},
            {"name": "skewness", "type": "Continuous Float", "unit": "Wavelet Skewness", "desc": "Skewness of Wavelet Transformed image.", "range": "-13.77 – 12.95"},
            {"name": "curtosis", "type": "Continuous Float", "unit": "Wavelet Kurtosis", "desc": "Kurtosis of Wavelet Transformed image.", "range": "-5.28 – 17.92"},
            {"name": "entropy", "type": "Continuous Float", "unit": "Entropy", "desc": "Information entropy of image.", "range": "-8.58 – 2.45"}
        ],
        "formula": {
            "title": "Linear Hyperplane & Maximum Margin Equation",
            "equation": "w · x + b = 0    |    Margin = 2 / ||w||",
            "components": [
                {"symbol": "w · x + b = 0", "meaning": "Equation of the linear decision boundary hyperplane"},
                {"symbol": "w", "meaning": "Weight vector perpendicular (orthogonal) to the hyperplane"},
                {"symbol": "b", "meaning": "Bias offset of the hyperplane from origin"},
                {"symbol": "2 / ||w||", "meaning": "Geometric margin distance between opposing support vectors"}
            ],
            "objective": "Maximize the margin width (2/||w||) while penalizing margin violations with parameter C."
        },
        "viva_qa": [
            {"q": "What is SVM?", "a": "Support Vector Machine is a supervised learning algorithm used for classification and regression."},
            {"q": "What is the main objective of SVM?", "a": "To find a decision boundary that separates classes with the maximum possible margin."},
            {"q": "What is a hyperplane?", "a": "A decision boundary that separates classes in the feature space."},
            {"q": "What are support vectors?", "a": "They are the training points closest to the decision boundary. They play an important role in determining the boundary."},
            {"q": "What is margin?", "a": "The distance between the decision boundary and the closest training points from the classes."},
            {"q": "What is Linear SVM?", "a": "An SVM that uses a linear decision boundary to separate classes."},
            {"q": "Which code is used?", "a": "from sklearn.svm import SVC; model = SVC(kernel='linear')"},
            {"q": "What is kernel='linear'?", "a": "It tells SVC to use a linear kernel to create a straight decision boundary."},
            {"q": "What is C in SVM?", "a": "C is a regularization parameter that controls the trade-off between a wider margin and classification errors on training data."},
            {"q": "Why do we use StandardScaler in SVM?", "a": "To put features on comparable scales, which is important for distance- and margin-based calculations."},
            {"q": "What happens when C is very high?", "a": "The model penalizes training errors more strongly, which can lead to a narrower margin and possible overfitting."},
            {"q": "What happens when C is small?", "a": "The model allows more training errors in exchange for stronger regularization and a potentially wider margin."}
        ],
        "conclusion": "Linear SVM achieved outstanding classification accuracy (>98%) on banknote authentication by identifying boundary support vectors."
    },

    "exp11": {
        "problem_statement": "Non-Linear Support Vector Machine uses the Radial Basis Function (RBF) kernel to learn nonlinear decision boundaries when classes cannot be separated by a straight hyperplane in original feature space.",
        "dataset_name": "data_banknote_authentication.csv (RBF Kernel Mapping)",
        "features": [
            {"name": "variance", "type": "Continuous Float", "unit": "Wavelet Variance", "desc": "Variance of Wavelet Transformed banknote image.", "range": "-7.04 – 6.82"},
            {"name": "skewness", "type": "Continuous Float", "unit": "Wavelet Skewness", "desc": "Skewness of Wavelet Transformed image.", "range": "-13.77 – 12.95"},
            {"name": "curtosis", "type": "Continuous Float", "unit": "Wavelet Kurtosis", "desc": "Kurtosis of Wavelet Transformed image.", "range": "-5.28 – 17.92"},
            {"name": "entropy", "type": "Continuous Float", "unit": "Entropy", "desc": "Information entropy of image.", "range": "-8.58 – 2.45"}
        ],
        "formula": {
            "title": "Radial Basis Function (RBF) Kernel",
            "equation": "K(x, x') = exp(-γ · ||x - x'||²)",
            "components": [
                {"symbol": "K(x, x')", "meaning": "RBF kernel similarity between two data points x and x'"},
                {"symbol": "||x - x'||²", "meaning": "Squared Euclidean distance between points in input space"},
                {"symbol": "γ (gamma)", "meaning": "Kernel coefficient controlling how far the influence of a single training point reaches"}
            ],
            "objective": "Enable nonlinear boundary separation without explicit high-dimensional coordinate mapping."
        },
        "viva_qa": [
            {"q": "What is Non-Linear SVM?", "a": "It is an SVM that uses a kernel to learn nonlinear decision boundaries."},
            {"q": "Why do we need Non-Linear SVM?", "a": "When the classes cannot be separated well using a straight-line boundary in the original feature space."},
            {"q": "What is a kernel?", "a": "A kernel function allows SVM to model relationships that can produce nonlinear decision boundaries without explicitly calculating coordinates in a higher-dimensional feature space."},
            {"q": "What is RBF?", "a": "RBF stands for Radial Basis Function."},
            {"q": "Which code is used?", "a": "model = SVC(kernel='rbf')"},
            {"q": "What is the difference between linear and RBF kernels?", "a": "Linear kernel produces a linear boundary for linearly separable patterns (kernel='linear'). RBF kernel can create nonlinear boundaries for complex patterns (kernel='rbf')."},
            {"q": "What is gamma in RBF?", "a": "Gamma controls how far the influence of an individual training point reaches."},
            {"q": "What happens when gamma is high?", "a": "Each training point has a more localized influence, which can create a complex decision boundary and increase overfitting risk."},
            {"q": "What happens when gamma is low?", "a": "Each point has a wider influence, which creates a smoother decision boundary and may increase underfitting risk."},
            {"q": "Why might blue and red points overlap in an SVM graph?", "a": "Because the features may not perfectly separate the classes. A model can still find a boundary that minimizes classification errors, depending on its parameters."},
            {"q": "Can RBF SVM classify overlapping data?", "a": "Yes. It can classify overlapping data, but it cannot guarantee perfect separation."}
        ],
        "conclusion": "The RBF kernel mapped data into non-linear Hilbert space, achieving 100% test classification accuracy on the banknote authentication dataset."
    },

    "exp12": {
        "problem_statement": "Random Forest Classifier is an ensemble learning algorithm that combines multiple decision trees (e.g. 100 trees) using bootstrap sampling and majority voting to make more accurate and stable predictions.",
        "dataset_name": "diabetes.csv (Ensemble of 100 Trees)",
        "features": [
            {"name": "Pregnancies", "type": "Integer", "unit": "Count", "desc": "Times pregnant.", "range": "0 – 17"},
            {"name": "Glucose", "type": "Integer", "unit": "mg/dL", "desc": "Plasma glucose concentration.", "range": "0 – 199"},
            {"name": "BloodPressure", "type": "Integer", "unit": "mm Hg", "desc": "Diastolic blood pressure.", "range": "0 – 122"},
            {"name": "SkinThickness", "type": "Integer", "unit": "mm", "desc": "Triceps skin fold thickness.", "range": "0 – 99"},
            {"name": "Insulin", "type": "Integer", "unit": "mu U/ml", "desc": "2-Hour serum insulin.", "range": "0 – 846"},
            {"name": "BMI", "type": "Continuous Float", "unit": "kg/m²", "desc": "Body mass index.", "range": "0.0 – 67.1"},
            {"name": "DiabetesPedigreeFunction", "type": "Continuous Float", "unit": "Score", "desc": "Pedigree score.", "range": "0.078 – 2.42"},
            {"name": "Age", "type": "Integer", "unit": "Years", "desc": "Patient age.", "range": "21 – 81"}
        ],
        "formula": {
            "title": "Random Forest Majority Voting Equation",
            "equation": "ŷ = mode { Tree₁(x), Tree₂(x), ..., Tree_B(x) }",
            "components": [
                {"symbol": "ŷ", "meaning": "Final ensemble predicted class label"},
                {"symbol": "Tree_b(x)", "meaning": "Class prediction from individual decision tree b"},
                {"symbol": "B = 100", "meaning": "Total number of decision trees in the forest (n_estimators)"},
                {"symbol": "mode", "meaning": "Majority voting (the class receiving the most votes wins)"}
            ],
            "objective": "Combine 100 randomized trees to reduce variance and combat single-tree overfitting."
        },
        "viva_qa": [
            {"q": "What is Random Forest?", "a": "Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions."},
            {"q": "What is ensemble learning?", "a": "Ensemble learning combines multiple models to produce a stronger or more reliable prediction."},
            {"q": "How does Random Forest work?", "a": "1. It creates multiple training samples using random sampling. 2. It trains multiple decision trees. 3. Each tree makes a prediction. 4. The final classification is selected using majority voting."},
            {"q": "Which class is used?", "a": "from sklearn.ensemble import RandomForestClassifier"},
            {"q": "What is n_estimators=100?", "a": "It means the Random Forest contains 100 decision trees."},
            {"q": "What is bootstrap sampling?", "a": "It is random sampling with replacement. Some observations may appear multiple times in a sample, while others may not appear."},
            {"q": "What is majority voting?", "a": "Each tree predicts a class, and the class receiving the most votes becomes the final prediction."},
            {"q": "What is feature importance?", "a": "It indicates how useful a feature was for making predictions across the trees."},
            {"q": "Why is Random Forest better than a single Decision Tree in many cases?", "a": "Combining multiple trees can reduce variance and improve generalization."},
            {"q": "What are the advantages of Random Forest?", "a": "Handles nonlinear relationships, often performs well on tabular data, less prone to overfitting than a single deep decision tree, can estimate feature importance."},
            {"q": "What are its disadvantages?", "a": "Requires more memory and computation than one tree, less interpretable than a single Decision Tree, can be slower to train."},
            {"q": "Does Random Forest require feature scaling?", "a": "Usually, no. Tree-based algorithms generally do not require StandardScaler."}
        ],
        "conclusion": "Averaging 100 decorrelated trees reduced test classification variance over an individual decision tree on the Diabetes diagnostic task."
    },

    "exp13": {
        "problem_statement": "Random Forest Regression is an ensemble algorithm that combines multiple decision trees to predict continuous numerical values by averaging individual tree forecasts.",
        "dataset_name": "diabetes.csv (Metabolic Biomarkers → Fasting Glucose mg/dL)",
        "features": [
            {"name": "Pregnancies", "type": "Integer", "unit": "Count", "desc": "Number of times pregnant.", "range": "0 – 17"},
            {"name": "BloodPressure", "type": "Integer", "unit": "mm Hg", "desc": "Diastolic blood pressure.", "range": "0 – 122"},
            {"name": "SkinThickness", "type": "Integer", "unit": "mm", "desc": "Triceps skin fold thickness.", "range": "0 – 99"},
            {"name": "Insulin", "type": "Integer", "unit": "mu U/ml", "desc": "2-Hour serum insulin.", "range": "0 – 846"},
            {"name": "BMI", "type": "Continuous Float", "unit": "kg/m²", "desc": "Body mass index.", "range": "0.0 – 67.1"},
            {"name": "DiabetesPedigreeFunction", "type": "Continuous Float", "unit": "Score", "desc": "Pedigree genetic score.", "range": "0.078 – 2.42"},
            {"name": "Age", "type": "Integer", "unit": "Years", "desc": "Patient age.", "range": "21 – 81"},
            {"name": "Glucose (Target)", "type": "Continuous Float", "unit": "mg/dL", "desc": "Plasma fasting glucose level.", "range": "44 – 199 mg/dL"}
        ],
        "formula": {
            "title": "Random Forest Ensemble Averaging Equation",
            "equation": "ŷ = (1 / B) · Σ_{b=1}^{B} Tree_b(x)",
            "components": [
                {"symbol": "ŷ", "meaning": "Final continuous numerical predicted output"},
                {"symbol": "Tree_b(x)", "meaning": "Numerical continuous prediction from individual regression tree b"},
                {"symbol": "B = 100", "meaning": "Total number of regression trees in the forest (n_estimators)"},
                {"symbol": "Averaging", "meaning": "Arithmetic mean of continuous predictions across all trees (no voting)"}
            ],
            "objective": "Mitigate high regression variance by averaging forecasts across 100 randomized trees."
        },
        "viva_qa": [
            {"q": "What is Random Forest Regression?", "a": "It is an ensemble algorithm that combines multiple decision trees to predict continuous numerical values."},
            {"q": "Which class is used?", "a": "from sklearn.ensemble import RandomForestRegressor"},
            {"q": "How does it work?", "a": "Each decision tree predicts a numerical value. The Random Forest calculates the average of the tree predictions to produce the final result."},
            {"q": "What is the difference between RandomForestClassifier and RandomForestRegressor?", "a": "RandomForestClassifier predicts categories and uses majority voting (e.g. Pass/Fail). RandomForestRegressor predicts numerical values and uses averaging (e.g. House price, Glucose)."},
            {"q": "What is n_estimators?", "a": "It specifies the number of decision trees in the forest."},
            {"q": "Which evaluation metrics are used?", "a": "Mean Squared Error (MSE), R² score, Mean Absolute Error (MAE)."},
            {"q": "Does Random Forest Regression use majority voting?", "a": "No. It averages the predictions of the individual trees."},
            {"q": "Can Random Forest Regression model nonlinear relationships?", "a": "Yes. It can model nonlinear patterns and interactions between features."},
            {"q": "Give an example.", "a": "Predicting house prices based on area, number of bedrooms, and location-related features."}
        ],
        "conclusion": "The Random Forest Regressor captured continuous non-linear interactions across metabolic biomarkers, predicting fasting glucose levels with low mean absolute error."
    }
}

def get_experiment_deep_dive(exp_id):
    """Returns deep-dive academic information for a specific experiment."""
    return EXPERIMENT_DEEP_DIVE.get(exp_id, {
        "problem_statement": "Experimental practical investigating supervised machine learning algorithms.",
        "dataset_name": "Standard Academic Laboratory Dataset",
        "features": [],
        "formula": {
            "title": "Algorithm Formula",
            "equation": "Algorithm Specific Equation",
            "components": [],
            "objective": "Minimize loss on training data."
        },
        "viva_qa": [],
        "conclusion": "Model trained and validated according to laboratory specifications."
    })
