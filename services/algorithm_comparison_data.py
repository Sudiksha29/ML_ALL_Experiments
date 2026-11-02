"""
services/algorithm_comparison_data.py
Comprehensive theoretical data and metadata for the 12 Machine Learning Predictive
Algorithms implemented in ML LabVerse, plus the separate Dataset Exploration tool.
"""

ALGORITHMS = [
    {
        "id": "simple_linear_regression",
        "name": "Simple Linear Regression",
        "short_name": "Simple Linear Reg",
        "type": "Regression",
        "type_badge": "Supervised Regression",
        "category_color": "blue",
        "icon": "trending-up",
        "exp_num": "EXP 02",
        "exp_id": "exp02",
        "learning_approach": "Parametric / Ordinary Least Squares (OLS)",
        "input_type": "1 Continuous Numerical Feature",
        "output_type": "Continuous Numerical Forecast",
        "formula": "y = mx + c",
        "formula_expanded": "\\hat{y} = \\beta_0 + \\beta_1 x_1",
        "scaling_required": "No (Single feature slope scales directly)",
        "scaling_badge": "Not Required",
        "interpretability": "Very High",
        "interpretability_score": 5,
        "complexity": "Low (O(n) closed-form OLS solution)",
        "summary": "Fits a straight line through two-dimensional data points by minimizing the sum of squared vertical residuals between actual and predicted target values.",
        "important_features": "One independent variable (X); predicts a continuous target (y); fits a straight line y = mx + c via closed-form OLS or gradient descent.",
        "advantages": [
            "Extremely simple, transparent, and computationally fast.",
            "Coefficients have direct physical interpretation (slope m indicates rate of change per unit X).",
            "No hyperparameter tuning required; has an analytical closed-form solution.",
            "Serves as the foundational baseline for all continuous numerical modeling."
        ],
        "limitations": [
            "Strictly restricted to modeling linear bivariate relationships.",
            "Highly sensitive to outliers, which disproportionately pull the regression line.",
            "Prone to severe underfitting if the underlying relationship is polynomial or periodic."
        ],
        "best_use_cases": "Baseline regression benchmarking, direct bivariate trend modeling (e.g., CGPA vs Salary package, Study Hours vs Exam Score, Temperature vs Energy demand).",
        "example_in_lab": "EXP 02: Predicting student placement package (LPA) from undergraduate CGPA, and final exam score from weekly study hours.",
        "key_parameters": [
            {"param": "fit_intercept", "desc": "Whether to calculate the intercept c (beta_0) for this model."},
            {"param": "copy_X", "desc": "If True, X will be copied; else it may be overwritten during calculation."}
        ],
        "how_it_works": "Calculates the centroid (mean_x, mean_y) and determines slope m = Cov(X,y) / Var(X). Computes intercept c = mean_y - m * mean_x to minimize Mean Squared Error (MSE)."
    },
    {
        "id": "multiple_linear_regression",
        "name": "Multiple Linear Regression",
        "short_name": "Multiple Linear Reg",
        "type": "Regression",
        "type_badge": "Supervised Regression",
        "category_color": "blue",
        "icon": "layers",
        "exp_num": "EXP 03",
        "exp_id": "exp03",
        "learning_approach": "Parametric / Matrix OLS (Normal Equation)",
        "input_type": "2 or More Numerical Features",
        "output_type": "Continuous Numerical Forecast",
        "formula": "y = b_0 + b_1 x_1 + b_2 x_2 + ... + b_n x_n",
        "formula_expanded": "\\hat{y} = \\beta^T X = (X^T X)^{-1} X^T y",
        "scaling_required": "Recommended (Required for Ridge/Lasso or Gradient Descent)",
        "scaling_badge": "Recommended",
        "interpretability": "High",
        "interpretability_score": 4,
        "complexity": "Moderate (O(p^2 n + p^3) matrix inversion)",
        "summary": "Extends linear regression to multidimensional space, fitting a hyper-plane across multiple independent predictors to estimate a continuous response.",
        "important_features": "Uses two or more independent variables; learns a dedicated partial regression coefficient b_i for each feature, holding other variables constant.",
        "advantages": [
            "Captures simultaneous multivariable relationships and partial covariate effects.",
            "Each coefficient reflects feature impact when controlling for other variables.",
            "Analytical solution via the Normal Equation (X^T X)^(-1) X^T y."
        ],
        "limitations": [
            "Assumes additive, strictly linear relationships across all dimensions.",
            "Multicollinearity (correlated features) inflates coefficient variance and destabilizes estimates.",
            "Sensitive to outliers and extreme leverage data points in multidimensional space."
        ],
        "best_use_cases": "Multivariate econometric forecasting, real estate house pricing, student placement prediction using both academic metrics and aptitude.",
        "example_in_lab": "EXP 03: Predicting engineering campus placement package using dual inputs: CGPA and IQ test score.",
        "key_parameters": [
            {"param": "fit_intercept", "desc": "Calculates hyper-plane offset b_0."},
            {"param": "positive", "desc": "When set to True, forces coefficients to be non-negative."}
        ],
        "how_it_works": "Solves the Normal Equation beta = (X^T X)^(-1) X^T y to find the coefficients that minimize sum of squared residuals in n-dimensional Euclidean space."
    },
    {
        "id": "binary_logistic_regression",
        "name": "Binary Logistic Regression",
        "short_name": "Binary Logistic Reg",
        "type": "Classification",
        "type_badge": "Supervised Binary Classification",
        "category_color": "purple",
        "icon": "toggle-left",
        "exp_num": "EXP 04",
        "exp_id": "exp04",
        "learning_approach": "Parametric / Maximum Likelihood Estimation (MLE)",
        "input_type": "Numerical or Encoded Tabular Features",
        "output_type": "Binary Class Label (0 or 1) & Probability [0, 1]",
        "formula": "P(y=1|x) = 1 / (1 + e^-(w*x + b))",
        "formula_expanded": "\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad z = w^T x + b",
        "scaling_required": "Yes (Required for regularized solvers like lbfgs)",
        "scaling_badge": "Crucial",
        "interpretability": "Very High",
        "interpretability_score": 5,
        "complexity": "Low to Moderate (O(n * p) iterative optimization)",
        "summary": "Applies the non-linear Sigmoid activation function to a linear combination of features, transforming real numbers into calibrated probabilities between 0 and 1.",
        "important_features": "Classification into two discrete classes; passes linear score z through logistic function; provides class probabilities and log-odds.",
        "advantages": [
            "Outputs genuine posterior class probabilities, not merely hard binary votes.",
            "Coefficients can be converted into meaningful Odds Ratios (exp(beta)).",
            "Very fast to train and evaluate, lightweight memory footprint, low risk of overfitting when regularized."
        ],
        "limitations": [
            "Establishes a strictly linear decision boundary in the input feature space.",
            "Struggles with non-linear or intricate geometric class boundaries without explicit feature interactions.",
            "Requires numeric encoding and feature scaling for regularized solvers."
        ],
        "best_use_cases": "Binary medical diagnosis (disease positive/negative), credit default risk, spam detection, Titanic passenger survival estimation.",
        "example_in_lab": "EXP 04: Titanic disaster passenger survival classification based on Passenger Class, Sex, Age, and Fare paid.",
        "key_parameters": [
            {"param": "C", "desc": "Inverse of regularization strength; smaller values specify stronger regularization."},
            {"param": "penalty", "desc": "Regularization norm ('l2', 'l1', or 'elasticnet')."},
            {"param": "solver", "desc": "Optimization algorithm (e.g., 'lbfgs', 'liblinear')."}
        ],
        "how_it_works": "Maps linear dot product w·x + b to probability interval (0, 1) using Sigmoid function. Minimizes Binary Cross-Entropy (Log-Loss) via gradient descent or L-BFGS."
    },
    {
        "id": "multiclass_logistic_regression",
        "name": "Multiclass Logistic Regression",
        "short_name": "Multiclass Logistic",
        "type": "Classification",
        "type_badge": "Supervised Multiclass",
        "category_color": "purple",
        "icon": "split",
        "exp_num": "EXP 05",
        "exp_id": "exp05",
        "learning_approach": "Parametric / Softmax Multinomial Loss",
        "input_type": "Numerical or Encoded Tabular Features",
        "output_type": "Multiclass Category & Probability Distribution",
        "formula": "P(y=k|x) = e^(w_k*x) / sum_j(e^(w_j*x))",
        "formula_expanded": "\\text{Softmax}(z_k) = \\frac{e^{z_k}}{\\sum_{j=1}^K e^{z_j}}",
        "scaling_required": "Yes (Crucial for gradient convergence across classes)",
        "scaling_badge": "Crucial",
        "interpretability": "High",
        "interpretability_score": 4,
        "complexity": "Moderate (O(K * n * p) where K is number of classes)",
        "summary": "Generalizes logistic regression to three or more categories using the Softmax function or One-vs-Rest (OvR) scheme, outputting a normalized probability distribution across classes.",
        "important_features": "Predicts among three or more classes; multinomial configuration jointly estimates class probability distribution summing to exactly 1.0.",
        "advantages": [
            "Calibrated multi-class probability distribution across all potential categories.",
            "Interpretable weights for each individual class outcome relative to the baseline.",
            "Standard, reliable linear baseline for multiclass tabular and categorical tasks."
        ],
        "limitations": [
            "Decision boundaries between any pair of classes remain planar / linear.",
            "Sensitive to feature scaling and imbalanced class representations.",
            "Requires significant feature engineering if underlying cluster boundaries are non-linear."
        ],
        "best_use_cases": "Customer tier categorization, multi-level academic grade prediction, Iris species classification, student performance tiering.",
        "example_in_lab": "EXP 05: Categorizing student academic performance into 3 classes (Low, Medium, High) based on study hours and lecture attendance.",
        "key_parameters": [
            {"param": "multi_class", "desc": "'multinomial' for joint Softmax or 'ovr' for One-vs-Rest."},
            {"param": "C", "desc": "Regularization strength parameter preventing overfitting on any single class."},
            {"param": "solver", "desc": "Must support multinomial loss (e.g. 'lbfgs', 'saga')."}
        ],
        "how_it_works": "Computes raw logits z_k for each class k, then applies Softmax normalization so sum of probabilities across all K classes equals 1.0."
    },
    {
        "id": "decision_tree",
        "name": "Decision Tree Classification",
        "short_name": "Decision Tree",
        "type": "Classification",
        "type_badge": "Supervised Non-Parametric",
        "category_color": "amber",
        "icon": "git-pull-request",
        "exp_num": "EXP 06",
        "exp_id": "exp06",
        "learning_approach": "Non-Parametric / Greedy Recursive Binary Splitting (CART)",
        "input_type": "Numerical, Categorical, Mixed Tabular Data",
        "output_type": "Class Label & Leaf Probability",
        "formula": "Gini = 1 - sum(p_i^2) | Entropy = -sum(p_i * log2(p_i))",
        "formula_expanded": "\\Delta I = I(\\text{parent}) - \\left(\\frac{N_L}{N} I(\\text{left}) + \\frac{N_R}{N} I(\\text{right})\\right)",
        "scaling_required": "No (Invariant to monotonic feature transformations)",
        "scaling_badge": "Not Required",
        "interpretability": "Very High (White-box)",
        "interpretability_score": 5,
        "complexity": "Low inference O(depth), Training O(n * p * log n)",
        "summary": "Builds a top-down hierarchical tree of intuitive IF-THEN decision rules by recursively selecting the feature split that maximizes Gini impurity reduction or Information Gain.",
        "important_features": "Hierarchical tree structure; splits data using criteria such as Gini impurity or Shannon Entropy; naturally models non-linear decision boundaries.",
        "advantages": [
            "Completely white-box model: easily explained, visualized, and audited by domain experts.",
            "Requires virtually zero data preprocessing: completely invariant to monotonic feature scaling.",
            "Naturally handles interactions between numerical and categorical features."
        ],
        "limitations": [
            "Extremely prone to overfitting if tree depth is unconstrained (memorizes training noise).",
            "High variance: small perturbations in the dataset can produce an entirely different tree structure.",
            "Forms axis-aligned rectangular decision boundaries, which can poorly approximate diagonal boundaries."
        ],
        "best_use_cases": "Clinical diagnostic flowcharts, regulatory compliance decisions where every choice must be audited, credit card underwriting.",
        "example_in_lab": "EXP 06: Clinical Heart Disease diagnosis (Present vs Absent) using 13 physiological indicators (blood pressure, cholesterol, ECG).",
        "key_parameters": [
            {"param": "max_depth", "desc": "The maximum depth of the tree; critical regularizer against overfitting."},
            {"param": "min_samples_split", "desc": "The minimum number of samples required to split an internal node."},
            {"param": "criterion", "desc": "'gini' for Gini impurity or 'entropy' for Information Gain."}
        ],
        "how_it_works": "Evaluates all candidate threshold splits across every feature, chooses the split maximizing impurity reduction Delta Gini, and recursively repeats until stopping criteria are satisfied."
    },
    {
        "id": "k_nearest_neighbors",
        "name": "K-Nearest Neighbors (KNN)",
        "short_name": "KNN Classifier",
        "type": "Classification",
        "type_badge": "Instance-Based / Lazy Learning",
        "category_color": "teal",
        "icon": "users",
        "exp_num": "EXP 07",
        "exp_id": "exp07",
        "learning_approach": "Non-Parametric / Instance-Based Memory Learning",
        "input_type": "Continuous Numerical Features (Scaled)",
        "output_type": "Class Label (Majority Vote) & Neighborhood Proportion",
        "formula": "d(x, y) = sqrt(sum((x_i - y_i)^2))",
        "formula_expanded": "\\hat{y} = \\text{mode}\\{y^{(1)}, y^{(2)}, \\dots, y^{(K)}\\}",
        "scaling_required": "Crucial (Distance metric is completely distorted by unscaled ranges)",
        "scaling_badge": "Crucial",
        "interpretability": "High",
        "interpretability_score": 4,
        "complexity": "Zero training O(1), High inference O(n * p) without KD-Tree",
        "summary": "A lazy learning algorithm that memorizes training instances without explicit training, predicting labels for query points through majority voting among its K nearest neighbors.",
        "important_features": "Predicts using the nearest K training samples; distance-based (Euclidean/Manhattan); lazy learner with zero model training phase.",
        "advantages": [
            "Extremely intuitive and natural: 'similar queries share similar outcomes'.",
            "Zero upfront training time; adapts instantly as new labelled observations arrive.",
            "Can learn highly flexible, arbitrary non-linear boundary shapes."
        ],
        "limitations": [
            "Computationally expensive inference: must compute distances against all training points for every prediction.",
            "Suffers severely from the Curse of Dimensionality in high-dimensional feature spaces.",
            "Crucially dependent on feature scaling (StandardScaler/MinMaxScaler) and sensitive to choice of K."
        ],
        "best_use_cases": "Local pattern matching, collaborative recommendation, biometric fingerprint/iris similarity, low-to-medium dimensional tabular tasks.",
        "example_in_lab": "EXP 07: Social Network Ads purchase prediction based on User Age and Estimated Annual Salary.",
        "key_parameters": [
            {"param": "n_neighbors", "desc": "Number of neighbors (K) to consider. Odd integers preferred to break ties."},
            {"param": "weights", "desc": "'uniform' (equal voting) or 'distance' (closer points have higher vote)."},
            {"param": "metric", "desc": "Distance metric: 'euclidean', 'manhattan', or 'minkowski'."}
        ],
        "how_it_works": "Calculates Euclidean distance from the query point to all training samples, sorts distances to identify the K closest neighbors, and assigns the majority class."
    },
    {
        "id": "gaussian_naive_bayes",
        "name": "Gaussian Naive Bayes",
        "short_name": "Gaussian NB",
        "type": "Classification",
        "type_badge": "Probabilistic Generative",
        "category_color": "emerald",
        "icon": "activity",
        "exp_num": "EXP 08",
        "exp_id": "exp08",
        "learning_approach": "Generative / Bayes Theorem with Gaussian Likelihood",
        "input_type": "Continuous Numerical Features",
        "output_type": "Class Label & Posterior Probability",
        "formula": "P(y|X) ~ P(y) * prod(P(x_i|y)) where x_i ~ N(mu, sigma^2)",
        "formula_expanded": "P(x_i | y) = \\frac{1}{\\sqrt{2\\pi\\sigma_{y}^2}} \\exp\\left(-\\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}\\right)",
        "scaling_required": "No (Mean and variance are computed independently per feature)",
        "scaling_badge": "Not Required",
        "interpretability": "High",
        "interpretability_score": 4,
        "complexity": "Very Low (O(n * p) one-pass summary statistics)",
        "summary": "Computes posterior class probabilities using Bayes' Theorem under the conditional independence assumption, estimating likelihoods via a Gaussian (Normal) bell curve.",
        "important_features": "Uses Bayes' theorem; assumes continuous numerical features follow a Gaussian distribution within each class; fast one-pass training.",
        "advantages": [
            "Blazing fast training and inference: requires only a single pass to compute mean and variance per feature.",
            "Performs surprisingly well even on small datasets where data is limited.",
            "Robust against irrelevant features; does not suffer from multicollinearity convergence failure."
        ],
        "limitations": [
            "Strong 'naive' assumption of conditional feature independence rarely holds in real-world data.",
            "If continuous features are heavily skewed or multi-modal, the Gaussian model poorly estimates likelihoods.",
            "Output probability values tend to be poorly calibrated (frequently pushed toward 0.0 or 1.0)."
        ],
        "best_use_cases": "Real-time streaming classification, sensor anomaly detection, numerical baseline classifier, small biological/medical diagnostic datasets.",
        "example_in_lab": "EXP 08: Diabetes risk prediction from 8 numerical clinical diagnostic indicators (Pima Indians Diabetes dataset).",
        "key_parameters": [
            {"param": "var_smoothing", "desc": "Portion of the largest variance of all features that is added to variances for calculation stability (default 1e-9)."}
        ],
        "how_it_works": "Calculates class priors P(y) and feature means mu_k and variances sigma_k^2 per class. For queries, evaluates Gaussian PDF for each feature, multiplies likelihoods, and picks max posterior."
    },
    {
        "id": "multinomial_naive_bayes",
        "name": "Multinomial Naive Bayes",
        "short_name": "Multinomial NB",
        "type": "Classification",
        "type_badge": "Supervised NLP / Text",
        "category_color": "emerald",
        "icon": "file-text",
        "exp_num": "EXP 09",
        "exp_id": "exp09",
        "learning_approach": "Generative / Discrete Count Likelihood with Laplace Smoothing",
        "input_type": "Discrete Word/Token Counts, TF-IDF Vectors",
        "output_type": "Class Label (e.g. Spam/Ham) & Log-Likelihoods",
        "formula": "P(w_i|c) = (N_ci + alpha) / (N_c + alpha * |V|)",
        "formula_expanded": "P(c | d) \\propto P(c) \\prod_{i=1}^M P(w_i | c)^{f_i}",
        "scaling_required": "No (Operates strictly on discrete non-negative count/frequency data)",
        "scaling_badge": "Not Required",
        "interpretability": "High",
        "interpretability_score": 4,
        "complexity": "Very Low (O(N_tokens) frequency table counting)",
        "summary": "Specialized probabilistic classifier designed for discrete count frequency data, famously utilized in Natural Language Processing and text classification via Bag-of-Words.",
        "important_features": "Uses word/token counts or non-negative frequencies; applies Bayes' theorem; incorporates Laplace smoothing (alpha=1.0) to prevent zero-probability traps.",
        "advantages": [
            "The industry benchmark baseline for text classification and spam filtering.",
            "Handles high-dimensional, sparse word-frequency matrices with millions of token vocabulary items effortlessly.",
            "Laplace smoothing elegantly handles unseen words without causing zero-multiplication collapse."
        ],
        "limitations": [
            "Completely ignores grammar, semantic word ordering, context, and negation ('not good' treated as 'not' + 'good').",
            "Cannot accept negative numbers; strictly demands non-negative count or frequency inputs.",
            "Conditional independence assumption fails when words strongly co-occur in phrases."
        ],
        "best_use_cases": "SMS spam detection, sentiment analysis of reviews, document topic categorization, email junk filtering.",
        "example_in_lab": "EXP 09: SMS Spam Classification (Spam vs Ham) using CountVectorizer tokenization and Laplace smoothing (alpha=1.0).",
        "key_parameters": [
            {"param": "alpha", "desc": "Additive (Laplace/Lidstone) smoothing parameter (1.0 for default Laplace)."},
            {"param": "fit_prior", "desc": "Whether to learn class prior probabilities or assume uniform priors."}
        ],
        "how_it_works": "Converts raw text into token count vectors with CountVectorizer. Estimates word frequencies per class with Laplace smoothing alpha=1.0, and assigns class with highest log-posterior."
    },
    {
        "id": "linear_svm",
        "name": "Linear Support Vector Machine",
        "short_name": "Linear SVM",
        "type": "Classification",
        "type_badge": "Supervised Maximum Margin",
        "category_color": "indigo",
        "icon": "shield-check",
        "exp_num": "EXP 10",
        "exp_id": "exp10",
        "learning_approach": "Convex Quadratic Optimization / Maximum Margin Hyperplane",
        "input_type": "Continuous Numerical Features (Standardized)",
        "output_type": "Binary Class Label & Signed Distance to Margin",
        "formula": "w * x + b = 0 | Margin = 2 / ||w||",
        "formula_expanded": "\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^n \\xi_i, \\quad y_i(w^T x_i + b) \\ge 1 - \\xi_i",
        "scaling_required": "Crucial (Margin maximization is directly distorted by unscaled ranges)",
        "scaling_badge": "Crucial",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "complexity": "Moderate (O(n^2 * p) to O(n * p) with LinearSVC)",
        "summary": "Finds the optimal separating hyperplane that maximizes the geometric distance (margin) between the closest opposing data points, known as Support Vectors.",
        "important_features": "Finds a maximum-margin separating hyperplane; uses a linear kernel; margin width is 2 / ||w||; controlled by regularization parameter C.",
        "advantages": [
            "Maximal margin boundary guarantees strong theoretical generalization bounds.",
            "Very effective in high-dimensional feature spaces where features outnumber samples.",
            "Memory efficient: decision boundary depends solely on the sparse subset of Support Vectors."
        ],
        "limitations": [
            "Cannot model complex non-linear or circular decision boundaries without kernel transformation.",
            "Critically sensitive to feature scaling (StandardScaler is mandatory).",
            "Does not naturally provide class probabilities (requires expensive Platt scaling calibration)."
        ],
        "best_use_cases": "Linearly separable classification, genomic feature discrimination, high-dimensional text classification, counterfeit detection.",
        "example_in_lab": "EXP 10: Banknote Authentication (Genuine vs Forged) based on 4 continuous wavelet transform statistical features.",
        "key_parameters": [
            {"param": "C", "desc": "Regularization parameter balancing margin width against margin violations (slack)."},
            {"param": "loss", "desc": "Loss function: 'hinge' or 'squared_hinge'."}
        ],
        "how_it_works": "Solves a quadratic programming problem to find normal vector w that maximizes margin 2/||w|| while penalizing misclassifications through slack variables xi_i weighted by C."
    },
    {
        "id": "non_linear_svm_rbf",
        "name": "Non-Linear SVM (RBF Kernel)",
        "short_name": "RBF Kernel SVM",
        "type": "Classification",
        "type_badge": "Supervised Kernel Method",
        "category_color": "indigo",
        "icon": "shield-alert",
        "exp_num": "EXP 11",
        "exp_id": "exp11",
        "learning_approach": "Kernel Method / Infinite-Dimensional Dual Quadratic Optimization",
        "input_type": "Continuous Numerical Features (Standardized)",
        "output_type": "Class Label & Decision Function Value",
        "formula": "K(x, x') = exp(-gamma * ||x - x'||^2)",
        "formula_expanded": "f(x) = \\text{sign}\\left( \\sum_{i \\in \\text{SV}} \\alpha_i y_i \\exp(-\\gamma \\|x_i - x\\|^2) + b \\right)",
        "scaling_required": "Crucial (Kernel Euclidean distances require standardization)",
        "scaling_badge": "Crucial",
        "interpretability": "Low (Geometric dual representation)",
        "interpretability_score": 2,
        "complexity": "High training O(n^2 to n^3), Inference O(#SV * p)",
        "summary": "Leverages the Kernel Trick via the Radial Basis Function (RBF) to implicitly map input vectors into an infinite-dimensional Hilbert space, separating non-linear patterns linearly.",
        "important_features": "Uses the RBF Gaussian kernel; models intricate non-linear decision boundaries without explicitly computing high-dimensional coordinates.",
        "advantages": [
            "Capable of capturing extremely complex, curved, multi-cluster decision boundaries.",
            "The Kernel Trick avoids the severe computational cost of high-dimensional coordinate transformation.",
            "Robust against moderate noise when properly tuned with cross-validation."
        ],
        "limitations": [
            "Requires careful joint hyperparameter tuning of both C and gamma.",
            "Computationally demanding for large datasets (training scales quadratically with sample size n).",
            "Low interpretability: the dual formulation behaves as an abstract geometric black box."
        ],
        "best_use_cases": "Non-linear pattern recognition, bioinformatics, complex signal classification, handwriting digit recognition.",
        "example_in_lab": "EXP 11: Non-linear Banknote Authentication under challenging overlapping boundary conditions.",
        "key_parameters": [
            {"param": "C", "desc": "Trade-off between smooth boundary and classifying training points correctly."},
            {"param": "gamma", "desc": "Kernel coefficient; defines how far the influence of a single training point reaches."},
            {"param": "kernel", "desc": "Specified as 'rbf' for Radial Basis Function."}
        ],
        "how_it_works": "Calculates pairwise Gaussian distances between samples using K(x, x') = exp(-gamma * ||x - x'||^2). Constructs an optimal separating hyperplane in the transformed Hilbert space."
    },
    {
        "id": "random_forest_classification",
        "name": "Random Forest Classification",
        "short_name": "Random Forest Class",
        "type": "Ensemble",
        "type_badge": "Ensemble Bagging Classification",
        "category_color": "rose",
        "icon": "trees",
        "exp_num": "EXP 12",
        "exp_id": "exp12",
        "learning_approach": "Ensemble Bagging (Bootstrap Aggregation) + Random Subspace",
        "input_type": "Mixed Tabular Data (Numerical, Categorical)",
        "output_type": "Class Label (Majority Vote) & Ensemble Probability",
        "formula": "y_hat = mode{Tree_1(x), Tree_2(x), ..., Tree_B(x)}",
        "formula_expanded": "P(y=k|x) = \\frac{1}{B} \\sum_{b=1}^B I(\\text{Tree}_b(x) = k)",
        "scaling_required": "No (Inherits scale invariance from underlying decision trees)",
        "scaling_badge": "Not Required",
        "interpretability": "Moderate (Gini Feature Importances available)",
        "interpretability_score": 3,
        "complexity": "Moderate to High O(B * n * p * log n), parallelizable",
        "summary": "Constructs an ensemble forest of de-correlated decision trees trained on bootstrap samples with random feature subsets, aggregating predictions via majority voting.",
        "important_features": "Combines many decision trees (typically 100); uses bootstrap sampling and random feature sub-sampling; mitigates variance through ensemble averaging.",
        "advantages": [
            "Substantially reduces variance and resists overfitting far better than a single decision tree.",
            "Produces high out-of-the-box accuracy across almost all tabular datasets.",
            "Naturally provides Mean Decrease in Impurity (MDI) feature importance rankings."
        ],
        "limitations": [
            "Significantly less interpretable than a single decision tree ('black-box ensemble').",
            "Requires more memory and computational resources for storing 100+ trees.",
            "Inference is slower than a single tree because all B trees must be evaluated."
        ],
        "best_use_cases": "Competitive tabular benchmarks, customer churn modeling, medical risk stratification, credit risk scoring.",
        "example_in_lab": "EXP 12: Diabetes onset classification using 100 bootstrapped estimators on Pima Indians clinical features.",
        "key_parameters": [
            {"param": "n_estimators", "desc": "Number of trees in the forest (e.g. 100). More trees reduce variance without overfitting."},
            {"param": "max_features", "desc": "Number of features to consider when looking for the best split (usually sqrt(n_features))."},
            {"param": "random_state", "desc": "Ensures reproducible bootstrap sampling and feature selection."}
        ],
        "how_it_works": "Draws B bootstrap samples with replacement from the training dataset. Trains an unpruned tree on each sample choosing among random feature subsets at every split. Aggregates via majority vote."
    },
    {
        "id": "random_forest_regression",
        "name": "Random Forest Regression",
        "short_name": "Random Forest Reg",
        "type": "Ensemble",
        "type_badge": "Ensemble Bagging Regression",
        "category_color": "rose",
        "icon": "bar-chart-3",
        "exp_num": "EXP 13",
        "exp_id": "exp13",
        "learning_approach": "Ensemble Bagging (Bootstrap Aggregation) + Arithmetic Averaging",
        "input_type": "Mixed Tabular Numerical Features",
        "output_type": "Continuous Numerical Forecast (Averaged)",
        "formula": "y_hat = (1 / B) * sum(Tree_b(x)) for b=1..B",
        "formula_expanded": "\\hat{y} = \\frac{1}{B} \\sum_{b=1}^B \\text{Tree}_b(x)",
        "scaling_required": "No (Decision trees split on individual feature ordinality)",
        "scaling_badge": "Not Required",
        "interpretability": "Moderate (Impurity-based Feature Importance)",
        "interpretability_score": 3,
        "complexity": "Moderate to High O(B * n * p * log n), parallelizable",
        "summary": "Ensemble regressor combining multiple randomized regression trees, averaging their individual continuous outputs to produce smooth, non-linear predictions with low variance.",
        "important_features": "Combines decision trees; averages continuous numerical predictions; captures complex non-linear feature interactions without feature scaling.",
        "advantages": [
            "Captures non-linear relationships and high-order feature interactions effortlessly.",
            "Substantially reduces prediction variance compared to an individual regression tree.",
            "Scale-invariant; insensitive to outliers in input feature distributions."
        ],
        "limitations": [
            "Cannot extrapolate outside the numerical range of the target training set (cannot predict values higher than maximum y_train).",
            "Model file size can be large when storing hundreds of deep regression trees.",
            "Slower inference than closed-form linear models."
        ],
        "best_use_cases": "Non-linear property appraisal, clinical glucose forecasting, energy grid load prediction, financial algorithmic pricing.",
        "example_in_lab": "EXP 13: Predicting continuous clinical blood glucose concentration (mg/dL) from metabolic metrics.",
        "key_parameters": [
            {"param": "n_estimators", "desc": "Number of trees in the forest (default 100)."},
            {"param": "criterion", "desc": "'squared_error' for MSE reduction or 'absolute_error' for MAE."},
            {"param": "max_depth", "desc": "Limits depth to control model storage and speed up inference."}
        ],
        "how_it_works": "Builds B regression trees with bootstrap sampling. For an unseen query, queries all B trees for continuous leaf outputs, and computes the arithmetic mean as the final forecast."
    }
]

# Separate entry for Dataset Exploration and Visualization (Analytical tool, not predictive ML)
DATA_EXPLORATION_TOOL = {
    "id": "dataset_exploration",
    "name": "Dataset Exploration & Visualization (EDA)",
    "type_badge": "Exploratory Data Analysis Tool",
    "exp_num": "EXP 01",
    "exp_id": "exp01",
    "is_predictive_algo": False,
    "description": "Dataset Exploration is a foundational data-analysis and diagnostic tool, NOT an automated predictive machine learning algorithm. It provides the analytical groundwork required before fitting any ML model.",
    "role_in_ml_pipeline": "Phase 1: Diagnostic Profiling & Preprocessing Assessment",
    "core_capabilities": [
        "Dataset shape profiling, column typing, and null-value auditing.",
        "Descriptive statistical summaries (mean, standard deviation, quartiles, IQR).",
        "Pearson correlation matrix computation to identify multicollinearity.",
        "Visual distribution histograms and bivariate scatter plots for exploratory trend detection.",
        "Custom CSV dataset uploading and dynamic feature inspection."
    ],
    "why_not_predictive": "EDA does not learn parameter weights, does not fit an objective loss function, and does not output predictions on unseen test samples. Instead, it informs which predictive algorithm should be chosen."
}

# Preset Side-by-Side Comparisons with detailed differential breakdowns
SIDE_BY_SIDE_PRESETS = [
    {
        "id": "linear_vs_logistic",
        "title": "Linear Regression vs Logistic Regression",
        "tag": "Regression vs Classification",
        "algo_ids": ["simple_linear_regression", "binary_logistic_regression"],
        "summary": "The quintessential comparison between continuous numerical forecasting and discrete categorical probability estimation.",
        "contrasts": [
            {"dimension": "Prediction Task", "algo_1": "Continuous numerical target (e.g. LPA, Salary)", "algo_2": "Discrete probability of binary class (0 or 1)"},
            {"dimension": "Mathematical Link", "algo_1": "Identity function: y = w·x + b", "algo_2": "Sigmoid activation: σ(z) = 1 / (1 + e^-z)"},
            {"dimension": "Output Range", "algo_1": "(-∞, +∞) unbounded real line", "algo_2": "[0, 1] bounded calibrated probability"},
            {"dimension": "Loss Function", "algo_1": "Mean Squared Error (MSE / L2 Loss)", "algo_2": "Binary Cross-Entropy (Log-Loss)"},
            {"dimension": "Primary Metric", "algo_1": "R² Score, RMSE, MAE", "algo_2": "Accuracy, Precision, Recall, ROC-AUC"},
            {"dimension": "Viva Takeaway", "algo_1": "Never use for classification because predictions exceed [0, 1]", "algo_2": "Converts linear score into valid odds ratio and probability"}
        ]
    },
    {
        "id": "slr_vs_mlr",
        "title": "Simple vs Multiple Linear Regression",
        "tag": "Dimensionality Scaling",
        "algo_ids": ["simple_linear_regression", "multiple_linear_regression"],
        "summary": "Contrasting single-variable bivariate slope fitting against multivariate n-dimensional hyper-plane estimation.",
        "contrasts": [
            {"dimension": "Predictor Count", "algo_1": "Exactly 1 independent variable (X)", "algo_2": "2 or more independent variables (X1, X2, ... Xp)"},
            {"dimension": "Geometric Nature", "algo_1": "2D straight line in a Cartesian plane", "algo_2": "Multidimensional hyper-plane in (p+1) space"},
            {"dimension": "Coefficient Meaning", "algo_1": "Total direct slope of Y with respect to X", "algo_2": "Partial regression coefficient holding other covariates fixed"},
            {"dimension": "Multicollinearity", "algo_1": "Impossible (only one feature exists)", "algo_2": "Critical vulnerability requiring VIF or correlation auditing"},
            {"dimension": "Computation", "algo_1": "Closed-form scalar Cov(X,y) / Var(X)", "algo_2": "Matrix inversion: (X^T X)^-1 X^T y"}
        ]
    },
    {
        "id": "binary_vs_multiclass",
        "title": "Binary vs Multiclass Logistic Regression",
        "tag": "Class Cardinality",
        "algo_ids": ["binary_logistic_regression", "multiclass_logistic_regression"],
        "summary": "Moving from two-class Sigmoid thresholding to multi-category Softmax probability normalization.",
        "contrasts": [
            {"dimension": "Target Classes", "algo_1": "Exactly 2 mutually exclusive classes (0 / 1)", "algo_2": "3 or more discrete categories (K >= 3)"},
            {"dimension": "Activation", "algo_1": "Sigmoid function σ(z)", "algo_2": "Softmax function: exp(z_k) / Σ exp(z_j)"},
            {"dimension": "Weight Vectors", "algo_1": "Single weight vector w of length p", "algo_2": "K distinct weight vectors (matrix of size K × p)"},
            {"dimension": "Loss Function", "algo_1": "Binary Cross-Entropy", "algo_2": "Categorical Cross-Entropy (Multinomial loss)"},
            {"dimension": "Alternative Architecture", "algo_1": "Not applicable", "algo_2": "One-vs-Rest (OvR) or One-vs-One (OvO)"}
        ]
    },
    {
        "id": "tree_vs_random_forest",
        "title": "Decision Tree vs Random Forest Classification",
        "tag": "Single Learner vs Ensemble",
        "algo_ids": ["decision_tree", "random_forest_classification"],
        "summary": "Comparing an interpretable single tree with high variance against a 100-tree bootstrapped ensemble with low variance.",
        "contrasts": [
            {"dimension": "Architecture", "algo_1": "Single greedy recursive decision tree", "algo_2": "Ensemble of 100 de-correlated bootstrap trees"},
            {"dimension": "Variance Sensitivity", "algo_1": "High (small data changes alter the whole tree)", "algo_2": "Low (averaging 100 trees cancels individual noise)"},
            {"dimension": "Overfitting Risk", "algo_1": "Very high if max_depth is unconstrained", "algo_2": "Significantly mitigated via bagging and feature subsampling"},
            {"dimension": "Interpretability", "algo_1": "High: Transparent IF-THEN rules (white-box)", "algo_2": "Moderate: Black-box voting with MDI feature importance"},
            {"dimension": "Inference Speed", "algo_1": "Ultra-fast: O(tree_depth) conditional jumps", "algo_2": "Moderate: Must query 100 separate tree models"},
            {"dimension": "Lab Observation", "algo_1": "EXP 06: Heart Disease accuracy ~72%", "algo_2": "EXP 12: Diabetes accuracy ~78% with higher F1"}
        ]
    },
    {
        "id": "gaussian_vs_multinomial_nb",
        "title": "Gaussian vs Multinomial Naive Bayes",
        "tag": "Continuous vs Discrete Text",
        "algo_ids": ["gaussian_naive_bayes", "multinomial_naive_bayes"],
        "summary": "Contrasting Gaussian continuous distribution modeling against discrete token count frequency estimation in NLP.",
        "contrasts": [
            {"dimension": "Feature Distribution", "algo_1": "Continuous real-valued numbers (~Normal N(μ, σ²))", "algo_2": "Discrete integer word counts, term frequencies"},
            {"dimension": "Likelihood Model", "algo_1": "Gaussian probability density function", "algo_2": "Multinomial distribution with Laplace smoothing"},
            {"dimension": "Input Representation", "algo_1": "Standard numerical tabular measurements", "algo_2": "CountVectorizer / Bag-of-Words sparse matrix"},
            {"dimension": "Zero-Frequency Problem", "algo_1": "Handled via variance smoothing (var_smoothing=1e-9)", "algo_2": "Handled via additive Laplace smoothing (α=1.0)"},
            {"dimension": "Primary Lab Domain", "algo_1": "EXP 08: Clinical diabetes diagnosis", "algo_2": "EXP 09: SMS Spam / Ham text detection"}
        ]
    },
    {
        "id": "linear_svm_vs_rbf_svm",
        "title": "Linear SVM vs Non-Linear RBF SVM",
        "tag": "Linear vs Non-Linear Manifold",
        "algo_ids": ["linear_svm", "non_linear_svm_rbf"],
        "summary": "Comparing a maximum-margin flat hyperplane against an infinite-dimensional Gaussian kernel mapping.",
        "contrasts": [
            {"dimension": "Kernel Function", "algo_1": "Linear: K(x, x') = x^T x'", "algo_2": "Radial Basis Function: K(x, x') = exp(-γ ||x - x'||²)"},
            {"dimension": "Decision Boundary", "algo_1": "Flat linear hyperplane: w·x + b = 0", "algo_2": "Curved, non-linear manifold in input space"},
            {"dimension": "Hyperparameters", "algo_1": "Regularization C only", "algo_2": "Regularization C and kernel coefficient gamma (γ)"},
            {"dimension": "Inference Latency", "algo_1": "Ultra-fast: single dot product w·x", "algo_2": "Moderate: computes kernel distance to all Support Vectors"},
            {"dimension": "Dimensionality Mapping", "algo_1": "Stays in original p-dimensional input space", "algo_2": "Maps vectors into infinite-dimensional Hilbert space"}
        ]
    },
    {
        "id": "knn_vs_decision_tree",
        "title": "K-Nearest Neighbors vs Decision Tree",
        "tag": "Lazy Instance vs Eager Tree",
        "algo_ids": ["k_nearest_neighbors", "decision_tree"],
        "summary": "Contrasting memory-based local neighborhood voting against global hierarchical partition rules.",
        "contrasts": [
            {"dimension": "Learning Paradigm", "algo_1": "Lazy Learner (Zero training, all work at query time)", "algo_2": "Eager Learner (Builds tree structure during training)"},
            {"dimension": "Feature Scaling", "algo_1": "Crucial (Euclidean distance requires StandardScaler)", "algo_2": "Not Required (Split rules depend on monotonic rank only)"},
            {"dimension": "Decision Boundaries", "algo_1": "Smooth, Voronoi-like polygonal boundaries", "algo_2": "Orthogonal, axis-aligned rectangular boxes"},
            {"dimension": "Prediction Latency", "algo_1": "Slow on large data: O(n * p) distance checks", "algo_2": "Very fast: O(depth) simple threshold checks"},
            {"dimension": "Dimensionality Risk", "algo_1": "Severe Curse of Dimensionality in high dimensions", "algo_2": "Naturally performs feature selection at each node"}
        ]
    },
    {
        "id": "rf_classifier_vs_regressor",
        "title": "Random Forest Classifier vs Regressor",
        "tag": "Voting vs Averaging",
        "algo_ids": ["random_forest_classification", "random_forest_regression"],
        "summary": "Contrasting ensemble majority voting for discrete classes against arithmetic mean averaging for continuous numbers.",
        "contrasts": [
            {"dimension": "Target Variable", "algo_1": "Discrete categorical class (e.g., Diabetic: Yes/No)", "algo_2": "Continuous numerical value (e.g., Glucose: mg/dL)"},
            {"dimension": "Ensemble Aggregation", "algo_1": "Majority voting / mode across all B trees", "algo_2": "Arithmetic mean (1/B) * Σ Tree_b(x)"},
            {"dimension": "Tree Split Metric", "algo_1": "Gini Impurity or Shannon Entropy reduction", "algo_2": "Mean Squared Error (MSE / Variance reduction)"},
            {"dimension": "Leaf Output", "algo_1": "Class distribution proportion / discrete label", "algo_2": "Mean numerical target of training samples in leaf"},
            {"dimension": "Extrapolation Constraint", "algo_1": "Bound to predefined categorical labels", "algo_2": "Cannot extrapolate outside range [min(y_train), max(y_train)]"}
        ]
    }
]

# Interactive Decision Tree Wizard data ("Which Algorithm Should I Choose?")
DECISION_WIZARD = {
    "goals": [
        {
            "id": "regression",
            "title": "Predict a Numerical Continuous Value",
            "subtitle": "E.g. Salary, Placement LPA, Blood Glucose, Housing Price",
            "icon": "trending-up",
            "recommendations": [
                {
                    "condition": "Single predictor, simple baseline, maximum transparency",
                    "algo_id": "simple_linear_regression",
                    "algo_name": "Simple Linear Regression",
                    "rationale": "Perfect when you have a single input feature and need a quick, highly interpretable baseline with an explicit formula (y = mx + c)."
                },
                {
                    "condition": "Multiple predictors, linear relationship, need partial effects",
                    "algo_id": "multiple_linear_regression",
                    "algo_name": "Multiple Linear Regression",
                    "rationale": "Ideal when combining multiple independent variables where each coefficient explains the feature's partial contribution holding others constant."
                },
                {
                    "condition": "Non-linear relationships, complex interactions, robust to outliers",
                    "algo_id": "random_forest_regression",
                    "algo_name": "Random Forest Regression",
                    "rationale": "The strongest choice for tabular numerical regression with complex non-linear patterns, high-order feature interactions, and zero scaling requirements."
                }
            ]
        },
        {
            "id": "binary_classification",
            "title": "Predict One of Two Categories (Binary)",
            "subtitle": "E.g. Survived/Died, Heart Disease Present/Absent, Genuine/Forged",
            "icon": "toggle-left",
            "recommendations": [
                {
                    "condition": "Need calibrated class probabilities and interpretable odds ratios",
                    "algo_id": "binary_logistic_regression",
                    "algo_name": "Binary Logistic Regression",
                    "rationale": "The foundational standard for binary classification; fast, interpretable, and produces genuine posterior probabilities."
                },
                {
                    "condition": "Need auditable IF-THEN decision rules with zero data scaling",
                    "algo_id": "decision_tree",
                    "algo_name": "Decision Tree Classifier",
                    "rationale": "Best when clinical or regulatory transparency requires visualizing the complete decision path leading to each diagnostic outcome."
                },
                {
                    "condition": "Instance-based local similarity, low dimensions, scaled data",
                    "algo_id": "k_nearest_neighbors",
                    "algo_name": "K-Nearest Neighbors (KNN)",
                    "rationale": "Natural for intuitive neighborhood-based decisions when data dimensionality is modest and features are standardized."
                },
                {
                    "condition": "High-dimensional feature space, approximately linearly separable",
                    "algo_id": "linear_svm",
                    "algo_name": "Linear SVM",
                    "rationale": "Optimal separating hyperplane maximizes geometric margin; excellent theoretical generalization in high dimensions."
                },
                {
                    "condition": "Complex curved boundaries, non-linear class separation",
                    "algo_id": "non_linear_svm_rbf",
                    "algo_name": "Non-Linear SVM (RBF Kernel)",
                    "rationale": "Uses the Gaussian kernel trick to separate highly overlapping, non-linear clusters in infinite-dimensional Hilbert space."
                },
                {
                    "condition": "Highest accuracy on tabular data, complex non-linear patterns",
                    "algo_id": "random_forest_classification",
                    "algo_name": "Random Forest Classifier",
                    "rationale": "Reduces variance of individual decision trees through 100-tree bagging; delivers top-tier performance on tabular datasets."
                }
            ]
        },
        {
            "id": "multiclass_classification",
            "title": "Predict One of Multiple Classes (3+)",
            "subtitle": "E.g. Student Performance (Low, Medium, High), Iris Species",
            "icon": "split",
            "recommendations": [
                {
                    "condition": "Linear multiclass baseline with Softmax normalized probabilities",
                    "algo_id": "multiclass_logistic_regression",
                    "algo_name": "Multiclass Logistic Regression",
                    "rationale": "Calibrated multi-class probability distributions summing to 1.0 via Softmax loss; highly interpretable class coefficients."
                },
                {
                    "condition": "Fast baseline on continuous clinical measurements",
                    "algo_id": "gaussian_naive_bayes",
                    "algo_name": "Gaussian Naive Bayes",
                    "rationale": "Extremely fast single-pass probability calculation assuming Normal feature distribution within each class."
                },
                {
                    "condition": "Hierarchical multi-branch categorization with zero preprocessing",
                    "algo_id": "decision_tree",
                    "algo_name": "Decision Tree Classifier",
                    "rationale": "Naturally partitions multi-class feature spaces into distinct leaf buckets without needing One-vs-Rest wrappers."
                },
                {
                    "condition": "Top predictive accuracy and stability across complex multi-class data",
                    "algo_id": "random_forest_classification",
                    "algo_name": "Random Forest Classifier",
                    "rationale": "Averages multi-class probability votes across 100 decorrelated trees; extremely robust to multi-class noise."
                }
            ]
        },
        {
            "id": "text_nlp",
            "title": "Categorize Text, Messages, or Documents",
            "subtitle": "E.g. Spam vs Ham, Sentiment Analysis, Topic Tagging",
            "icon": "file-text",
            "recommendations": [
                {
                    "condition": "Word count frequencies, sparse token vocabularies, Laplace smoothing",
                    "algo_id": "multinomial_naive_bayes",
                    "algo_name": "Multinomial Naive Bayes",
                    "rationale": "The definitive benchmark for text classification using Bag-of-Words and CountVectorizer token frequencies; blazingly fast and robust to sparse matrices."
                },
                {
                    "condition": "High-dimensional sparse text vectors, linearly separable",
                    "algo_id": "linear_svm",
                    "algo_name": "Linear SVM",
                    "rationale": "Frequently used alongside TF-IDF vectors in text mining because linear hyperplanes scale exceptionally well in high-dimensional vocabulary spaces."
                }
            ]
        }
    ]
}

# Curated Official Scikit-Learn Technical Documentation References
SCIKIT_LEARN_REFERENCES = [
    {
        "module": "Supervised Learning User Guide",
        "url": "https://scikit-learn.org/stable/user_guide.html",
        "description": "Comprehensive reference guide covering algorithm theory, mathematical formulation, and API best practices.",
        "badge": "Official Guide",
        "key_algorithms": "All 12 Scikit-Learn Estimators"
    },
    {
        "module": "Support Vector Machines (SVM)",
        "url": "https://scikit-learn.org/stable/modules/svm.html",
        "description": "Formulations for SVC, LinearSVC, NuSVC, and SVR. Details primal vs dual quadratic optimization, slack penalties (C), and kernel mathematical definitions (Linear, Polynomial, RBF).",
        "badge": "Kernel Theory",
        "key_algorithms": "Linear SVM (EXP 10), Non-Linear RBF SVM (EXP 11)"
    },
    {
        "module": "Decision Trees",
        "url": "https://scikit-learn.org/stable/modules/tree.html",
        "description": "Details the optimized CART (Classification and Regression Trees) implementation. Covers Gini impurity, Shannon Entropy, minimal cost-complexity pruning, and complexity bounds.",
        "badge": "Tree Induction",
        "key_algorithms": "Decision Tree Classifier (EXP 06)"
    },
    {
        "module": "Ensemble Methods",
        "url": "https://scikit-learn.org/stable/modules/ensemble.html",
        "description": "Covers bagging ensembles, random forest de-correlation, bootstrap sampling, out-of-bag (OOB) error estimation, and Mean Decrease in Impurity (MDI) feature importances.",
        "badge": "Ensemble Learning",
        "key_algorithms": "Random Forest Classifier (EXP 12), Random Forest Regressor (EXP 13)"
    },
    {
        "module": "Generalized Linear Models",
        "url": "https://scikit-learn.org/stable/modules/linear_model.html",
        "description": "Formulations for Ordinary Least Squares (LinearRegression) and LogisticRegression (L-BFGS, coordinate descent, Newton-CG, and multinomial Softmax cross-entropy loss).",
        "badge": "Linear Models",
        "key_algorithms": "Simple Linear (EXP 02), Multiple Linear (EXP 03), Binary Logistic (EXP 04), Multiclass Logistic (EXP 05)"
    },
    {
        "module": "Naive Bayes",
        "url": "https://scikit-learn.org/stable/modules/naive_bayes.html",
        "description": "Covers GaussianNB likelihood estimation via sample mean and variance, and MultinomialNB discrete count smoothing via additive Laplace/Lidstone smoothing (alpha).",
        "badge": "Bayesian Methods",
        "key_algorithms": "Gaussian Naive Bayes (EXP 08), Multinomial Naive Bayes (EXP 09)"
    },
    {
        "module": "Nearest Neighbors",
        "url": "https://scikit-learn.org/stable/modules/neighbors.html",
        "description": "Covers instance-based unsupervised and supervised nearest neighbors, BallTree and KDTree indexing structures for fast spatial queries, and distance weighting schemes.",
        "badge": "Instance-Based",
        "key_algorithms": "K-Nearest Neighbors (EXP 07)"
    }
]

def get_all_algorithms():
    """Return all 12 predictive algorithm dicts."""
    return ALGORITHMS

def get_algorithm_by_id(algo_id):
    """Retrieve algorithm data by string id."""
    for algo in ALGORITHMS:
        if algo["id"] == algo_id:
            return algo
    return None

def get_preset_by_id(preset_id):
    """Retrieve side-by-side comparison preset by id."""
    for preset in SIDE_BY_SIDE_PRESETS:
        if preset["id"] == preset_id:
            return preset
    return None
