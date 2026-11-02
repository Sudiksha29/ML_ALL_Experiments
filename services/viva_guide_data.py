"""
ML LabVerse — Complete Machine Learning Practical Viva Questions & Answers
Covering All 13 Topics, General Concepts, Comparisons, Code Questions, Formula Revision & Quick Test.
"""

COMMON_ML_CONCEPTS = [
    {"q": "What is Machine Learning?", "a": "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions without being explicitly programmed for every task."},
    {"q": "What are the main types of Machine Learning?", "a": "1. Supervised Learning – learns from labelled data.\n2. Unsupervised Learning – finds patterns in unlabelled data.\n3. Reinforcement Learning – learns through rewards and penalties."},
    {"q": "What is supervised learning?", "a": "It is a type of learning where the model is trained using input data and its correct output (label)."},
    {"q": "What is the difference between classification and regression?", "a": "Classification predicts discrete categories (e.g. Pass/Fail, evaluated via Accuracy, F1-score). Regression predicts continuous numerical values (e.g. Salary, Package, evaluated via MSE, R² score)."},
    {"q": "What are features and target variables?", "a": "Features (X) are input columns used for prediction. Target (y) is the output column that we want to predict."},
    {"q": "Explain this code: X = df.drop('target', axis=1); y = df['target']", "a": "X contains all input columns except the target. y contains only the target column. axis=1 means we are dropping a column."},
    {"q": "Why do we split the dataset?", "a": "To train the model on one part and test its performance on unseen data."},
    {"q": "What is train_test_split()?", "a": "It divides the dataset into training and testing portions."},
    {"q": "What is test_size=0.2?", "a": "It means 20% of the data is used for testing and 80% for training."},
    {"q": "What is random_state=42?", "a": "It makes the random splitting reproducible. The same seed gives the same split when the other conditions remain unchanged."},
    {"q": "What is overfitting?", "a": "When a model performs very well on training data but poorly on unseen data because it has learned the training data too closely."},
    {"q": "What is underfitting?", "a": "When a model is too simple to learn the important patterns in the data."},
    {"q": "What is preprocessing?", "a": "It is the process of preparing raw data before giving it to a machine learning algorithm. It may include handling missing values, encoding categories, and scaling features."},
    {"q": "What is fit()?", "a": "It trains the model by learning patterns from the training data."},
    {"q": "What is predict()?", "a": "It uses the trained model to predict outputs for new input data."},
    {"q": "What is the difference between fit() and fit_transform()?", "a": "fit() learns parameters from data. fit_transform() learns parameters and transforms the data in a single step."},
    {"q": "What is accuracy?", "a": "Accuracy is the proportion of correctly predicted observations out of all observations: (TP + TN) / (TP + TN + FP + FN)."},
    {"q": "What is a confusion matrix?", "a": "It is a table that compares actual and predicted classes using True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)."},
    {"q": "What is a .pkl or .joblib file?", "a": "It is used to serialize and save a trained Python model or Python objects so they can be loaded later without retraining."},
    {"q": "What is Scikit-learn?", "a": "Scikit-learn is a Python library used for machine learning, including preprocessing, training, prediction, and evaluation."}
]

COMPARISON_QUESTIONS = [
    {"q": "Difference between AI, ML, and Deep Learning?", "a": "AI: Making machines perform intelligent tasks. ML: Learning patterns from data. Deep Learning: A subset of ML that uses multilayer neural networks."},
    {"q": "Difference between supervised and unsupervised learning?", "a": "Supervised learning uses labelled data, while unsupervised learning works with unlabelled data."},
    {"q": "Difference between Linear and Logistic Regression?", "a": "Linear Regression is for regression, predicts continuous values, and uses a linear line equation. Logistic Regression is for classification, predicts class probabilities, and uses a logistic sigmoid function."},
    {"q": "Difference between SLR and MLR?", "a": "SLR uses one independent variable. MLR uses two or more independent variables."},
    {"q": "Difference between Logistic Regression and Decision Tree?", "a": "Logistic Regression learns a mathematical decision boundary, while a Decision Tree learns hierarchical decision rules."},
    {"q": "Difference between Decision Tree and Random Forest?", "a": "A Decision Tree uses one tree. Random Forest combines multiple trees using bootstrap bagging."},
    {"q": "Difference between KNN and Naive Bayes?", "a": "KNN uses distance and neighboring observations. Naive Bayes uses probability and Bayes' theorem."},
    {"q": "Difference between GaussianNB and MultinomialNB?", "a": "GaussianNB models continuous numerical features using a Gaussian distribution. MultinomialNB is commonly used for non-negative word-count features."},
    {"q": "Difference between Linear SVM and Non-Linear SVM?", "a": "Linear SVM uses a linear decision boundary, while Non-Linear SVM uses a kernel such as RBF to model nonlinear boundaries."},
    {"q": "Difference between KNN and SVM?", "a": "KNN classifies using nearby training examples, whereas SVM learns a decision boundary that maximizes the margin."},
    {"q": "Which algorithms require feature scaling in your practicals?", "a": "KNN and SVM are particularly sensitive to feature scales. StandardScaler is commonly used with them. Linear and Logistic Regression can also benefit from scaling."},
    {"q": "Which algorithms generally do not require feature scaling?", "a": "Decision Tree and Random Forest."},
    {"q": "Which algorithms are used for regression in your practicals?", "a": "Simple Linear Regression, Multiple Linear Regression, and Random Forest Regression."},
    {"q": "Which algorithms are used for classification in your practicals?", "a": "Binary Logistic Regression, Multiclass Logistic Regression, Decision Tree Classifier, KNN Classifier, Gaussian Naive Bayes, Multinomial Naive Bayes, Linear SVM, Non-Linear SVM, and Random Forest Classifier."},
    {"q": "What is the difference between bagging and boosting?", "a": "Bagging trains multiple models independently, often on bootstrap samples, and combines their predictions (Random Forest is based on bagging). Boosting trains models sequentially, with later models focusing on errors made by earlier models."},
    {"q": "Name some ensemble learning algorithms.", "a": "Random Forest, AdaBoost, Gradient Boosting, XGBoost, Voting Classifier, and Stacking."},
    {"q": "Which algorithm is best among all these?", "a": "There is no single best algorithm. It depends on the dataset, problem, performance, interpretability, and computational requirements."}
]

CODE_VIVA_QUESTIONS = [
    {"code": "pd.read_csv()", "desc": "Reads a CSV file into a Pandas DataFrame."},
    {"code": "df.head()", "desc": "Displays the first five rows by default."},
    {"code": "df.shape", "desc": "Returns a tuple representing (number of rows, number of columns)."},
    {"code": "df.columns", "desc": "Displays all column names in the DataFrame."},
    {"code": "df.isnull().sum()", "desc": "Counts missing (null/NaN) values for each column."},
    {"code": "train_test_split()", "desc": "Splits arrays or matrices into random train and test subsets."},
    {"code": "model.fit()", "desc": "Trains the model by learning parameters from training data."},
    {"code": "model.predict()", "desc": "Makes predictions on new inputs using the trained model parameters."},
    {"code": "accuracy_score()", "desc": "Calculates overall classification accuracy on true vs predicted labels."},
    {"code": "confusion_matrix()", "desc": "Builds a table comparing actual vs predicted classes (TP, TN, FP, FN)."},
    {"code": "classification_report()", "desc": "Displays precision, recall, and F1-score for each class."},
    {"code": "mean_squared_error()", "desc": "Calculates average squared difference between actual and predicted values."},
    {"code": "r2_score()", "desc": "Calculates R² coefficient of determination (variance explained)."},
    {"code": "StandardScaler()", "desc": "Standardizes features by removing the mean and scaling to unit variance."},
    {"code": "CountVectorizer()", "desc": "Converts a collection of text documents to a matrix of token counts."},
    {"code": "pickle.dump() / joblib.dump()", "desc": "Serializes and saves a trained Python model object to disk."},
    {"code": "pickle.load() / joblib.load()", "desc": "Loads a saved model pipeline object from disk."}
]

MUST_KNOW_CODE_QUESTIONS = [
    {"q": "What is the difference between fit() and predict()?", "a": "fit() trains the model by finding parameters from training data. predict() makes predictions using those learned parameters on new data."},
    {"q": "Why do we use axis=1?", "a": "Because axis=1 refers to columns. axis=0 refers to rows."},
    {"q": "Why do we use random_state=42?", "a": "To make random operations reproducible across runs."},
    {"q": "Why do we use StandardScaler()?", "a": "To standardize features so their numerical scales are comparable, preventing large-scale features from dominating distance calculations."},
    {"q": "Why do we save the model using Pickle/Joblib?", "a": "So that we can reuse the trained model in production or a web app without needing to re-train it every time."}
]

LAST_MINUTE_FORMULAS = [
    {"concept": "Simple Linear Regression", "formula": "y = mx + c"},
    {"concept": "Multiple Linear Regression", "formula": "y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ"},
    {"concept": "Sigmoid (Logistic)", "formula": "σ(z) = 1 / (1 + e⁻ᶻ)"},
    {"concept": "Softmax Function", "formula": "Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)"},
    {"concept": "Gini Impurity Index", "formula": "Gini = 1 - Σ (pᵢ)²"},
    {"concept": "Euclidean Distance", "formula": "d = √[ Σ (xᵢ - yᵢ)² ]"},
    {"concept": "Bayes' Theorem", "formula": "P(A|B) = [ P(B|A) · P(A) ] / P(B)"},
    {"concept": "Gaussian Likelihood", "formula": "P(xᵢ|y) = [ 1 / √(2πσ²) ] · e^[ -(xᵢ - μ)² / (2σ²) ]"},
    {"concept": "Laplace Smoothing", "formula": "P(wᵢ | c) = (N_ci + α) / (N_c + α·|V|)"},
    {"concept": "Linear SVM Hyperplane", "formula": "w · x + b = 0   |   Margin = 2 / ||w||"},
    {"concept": "RBF Kernel", "formula": "K(x, x') = exp(-γ · ||x - x'||²)"},
    {"concept": "Random Forest Classification", "formula": "ŷ = mode { Tree₁(x), ..., Tree_B(x) }"},
    {"concept": "Random Forest Regression", "formula": "ŷ = (1 / B) · Σ Tree_b(x)"},
    {"concept": "Mean Squared Error (MSE)", "formula": "MSE = (1 / n) · Σ (yᵢ - ŷᵢ)²"},
    {"concept": "Accuracy Score", "formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN)"},
    {"concept": "Precision Score", "formula": "Precision = TP / (TP + FP)"},
    {"concept": "Recall (Sensitivity)", "formula": "Recall = TP / (TP + FN)"},
    {"concept": "F1-Score", "formula": "F1 = 2 · (Precision · Recall) / (Precision + Recall)"}
]

TEST_YOUR_PREP_QUIZ = [
    {
        "id": 1,
        "q": "Which algorithm uses majority voting among multiple decision trees?",
        "options": ["KNN", "Random Forest Classifier", "Linear Regression", "GaussianNB"],
        "correct": "Random Forest Classifier",
        "explanation": "Random Forest combines multiple decision trees using bootstrap aggregation and majority voting."
    },
    {
        "id": 2,
        "q": "Which algorithm commonly uses CountVectorizer for text classification?",
        "options": ["GaussianNB", "Decision Tree", "MultinomialNB", "Linear SVM"],
        "correct": "MultinomialNB",
        "explanation": "MultinomialNB operates on discrete word counts generated by CountVectorizer."
    },
    {
        "id": 3,
        "q": "What does K represent in KNN?",
        "options": ["Number of classes", "Number of nearest neighbors", "Number of features", "Number of datasets"],
        "correct": "Number of nearest neighbors",
        "explanation": "K is the number of nearest data points considered to determine the class by voting."
    },
    {
        "id": 4,
        "q": "Which SVM kernel is commonly used for nonlinear boundaries?",
        "options": ["linear", "rbf", "none", "mean"],
        "correct": "rbf",
        "explanation": "The Radial Basis Function (rbf) kernel projects data into higher dimensions to separate nonlinear patterns."
    },
    {
        "id": 5,
        "q": "Which metric is commonly used for regression?",
        "options": ["Confusion matrix", "Accuracy", "Mean Squared Error", "Precision"],
        "correct": "Mean Squared Error",
        "explanation": "MSE measures average squared difference between continuous actual and predicted values."
    },
    {
        "id": 6,
        "q": "Which algorithm assumes numerical features follow a Gaussian distribution within each class?",
        "options": ["MultinomialNB", "GaussianNB", "KNN", "Random Forest"],
        "correct": "GaussianNB",
        "explanation": "GaussianNB estimates mean and variance per feature assuming a normal bell-shaped distribution."
    },
    {
        "id": 7,
        "q": "What does n_estimators=100 mean in Random Forest?",
        "options": ["100 features", "100 classes", "100 decision trees", "100 test samples"],
        "correct": "100 decision trees",
        "explanation": "n_estimators controls the number of randomized decision trees in the ensemble forest."
    },
    {
        "id": 8,
        "q": "Which algorithm uses a sigmoid function?",
        "options": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "correct": "Logistic Regression",
        "explanation": "Logistic Regression applies the Sigmoid function σ(z) = 1/(1+e^-z) to map scores into probabilities."
    }
]
