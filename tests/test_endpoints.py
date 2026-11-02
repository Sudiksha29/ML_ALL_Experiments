import os
import sys
import unittest
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
import config

class MLLabVerseTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_dashboard_routes(self):
        """Test main navigation pages return 200 OK"""
        routes = ["/", "/experiments", "/datasets", "/comparison", "/algorithm-comparison", "/learning-center", "/model-playground", "/profile", "/about", "/history"]
        for r in routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Route {r} failed with {res.status_code}")

    def test_algorithm_comparison_hub(self):
        """Test the dedicated Algorithm Comparison Hub page and all 12 algorithms"""
        res = self.client.get("/algorithm-comparison")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Verify key header and tab titles
        self.assertIn("Algorithm Comparison Hub", html)
        self.assertIn("Overview & Full Comparison Table", html)
        self.assertIn("Side-by-Side Comparator", html)
        self.assertIn("Which Algorithm Should I Choose?", html)
        self.assertIn("Official Scikit-Learn Technical Reference", html)

        # Verify Dataset Exploration is separated as an analytical tool
        self.assertIn("Analytical Tool • Not Predictive ML", html)
        self.assertIn("EXP 01 — Dataset Exploration", html)

        # Verify all 12 predictive algorithms are included
        expected_algorithms = [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Binary Logistic Regression",
            "Multiclass Logistic Regression",
            "Decision Tree Classification",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Multinomial Naive Bayes",
            "Linear Support Vector Machine",
            "Non-Linear SVM (RBF Kernel)",
            "Random Forest Classification",
            "Random Forest Regression"
        ]
        for algo in expected_algorithms:
            self.assertIn(algo, html, f"Algorithm '{algo}' missing from /algorithm-comparison")

        # Test API endpoints
        res_api = self.client.get("/api/algorithm-comparison")
        self.assertEqual(res_api.status_code, 200)
        data = res_api.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["algorithms"]), 12)
        self.assertEqual(len(data["presets"]), 8)

        # Test single algorithm API
        res_single = self.client.get("/api/algorithm-comparison/non_linear_svm_rbf")
        self.assertEqual(res_single.status_code, 200)
        single_data = res_single.get_json()
        self.assertEqual(single_data["status"], "success")
        self.assertEqual(single_data["algorithm"]["id"], "non_linear_svm_rbf")
        self.assertIn("exp", single_data["algorithm"]["formula"].lower())
        self.assertIn("rbf", single_data["algorithm"]["short_name"].lower())

    def test_all_13_experiment_pages(self):
        """Test that all 13 individual experiment detail pages render 200 OK with deep-dive tabs"""
        for exp_id in config.EXPERIMENTS_META.keys():
            res = self.client.get(f"/experiment/{exp_id}")
            self.assertEqual(res.status_code, 200, f"Exp page {exp_id} failed with {res.status_code}")
            html = res.get_data(as_text=True)
            self.assertIn("Problem & Objective", html, f"Tab Problem & Objective missing in {exp_id}")
            self.assertIn("Dataset & Feature Dictionary", html, f"Tab Dataset missing in {exp_id}")
            self.assertIn("Mathematical Formulation", html, f"Tab Math missing in {exp_id}")
            self.assertIn("Viva Examination Q&A", html, f"Tab Viva missing in {exp_id}")
            self.assertIn("Observations & Conclusion", html, f"Tab Conclusion missing in {exp_id}")

    def test_prediction_exp02_slr(self):
        """Test Simple Linear Regression inference (Placement & Hours)"""
        # Placement
        res = self.client.post("/api/predict/exp02", json={
            "switcher_id": "placement",
            "inputs": {"cgpa": 8.5}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("LPA", data["label"])

        # Study hours
        res = self.client.post("/api/predict/exp02", json={
            "switcher_id": "hours",
            "inputs": {"hours": 5.0}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Marks", data["label"])

    def test_prediction_exp03_mlr(self):
        """Test Multiple Linear Regression inference"""
        res = self.client.post("/api/predict/exp03", json={
            "switcher_id": "cgpa_iq",
            "inputs": {"cgpa": 8.0, "iq": 120}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("LPA", data["label"])

    def test_prediction_exp04_binary_logistic(self):
        """Test Binary Logistic Regression (Titanic)"""
        res = self.client.post("/api/predict/exp04", json={
            "switcher_id": "titanic",
            "inputs": {"pclass": 1, "sex": 1, "age": 28, "fare": 80.0}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Survived", data["label"])
        self.assertIsNotNone(data["confidence"])

    def test_prediction_exp05_multiclass_logistic(self):
        """Test Multiclass Logistic Regression (Student Performance)"""
        res = self.client.post("/api/predict/exp05", json={
            "switcher_id": "student",
            "inputs": {"study_hours": 7.0, "attendance": 80.0}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Performance", data["label"])

    def test_prediction_exp06_decision_tree(self):
        """Test Decision Tree on Heart Disease"""
        res = self.client.post("/api/predict/exp06", json={
            "inputs": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2,
                "slope": 1, "ca": 0, "thal": 2
            }
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn(data["prediction"], [0, 1])

    def test_prediction_exp07_knn(self):
        """Test KNN on Diabetes"""
        res = self.client.post("/api/predict/exp07", json={
            "switcher_id": "diabetes",
            "inputs": {
                "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70, "SkinThickness": 25,
                "Insulin": 100, "BMI": 31.5, "DiabetesPedigreeFunction": 0.45, "Age": 35
            }
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Diabetic", data["label"])

    def test_prediction_exp08_gaussian_nb(self):
        """Test Gaussian Naive Bayes on Banknote"""
        res = self.client.post("/api/predict/exp08", json={
            "inputs": {"variance": 2.1, "skewness": 5.4, "curtosis": -0.5, "entropy": -1.0}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Banknote", data["label"])

    def test_prediction_exp09_multinomial_nb(self):
        """Test Multinomial Naive Bayes Campus Message Detective"""
        res = self.client.post("/api/predict/exp09", json={
            "inputs": {"message": "Please submit your Machine Learning project report before 5 PM tomorrow."}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["prediction"], "Important")

    def test_prediction_exp10_linear_svm(self):
        """Test Linear SVM"""
        res = self.client.post("/api/predict/exp10", json={
            "inputs": {"variance": 3.2, "skewness": 7.1, "curtosis": -2.1, "entropy": -0.5}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Banknote", data["label"])

    def test_prediction_exp11_nonlinear_svm(self):
        """Test Non-Linear RBF SVM"""
        res = self.client.post("/api/predict/exp11", json={
            "inputs": {"variance": -1.2, "skewness": 3.1, "curtosis": 1.5, "entropy": -0.8}
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Banknote", data["label"])

    def test_prediction_exp12_rf_classifier(self):
        """Test Random Forest Classifier"""
        res = self.client.post("/api/predict/exp12", json={
            "inputs": {
                "Pregnancies": 4, "Glucose": 160, "BloodPressure": 80, "SkinThickness": 30,
                "Insulin": 150, "BMI": 34.0, "DiabetesPedigreeFunction": 0.65, "Age": 45
            }
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Diabetic", data["label"])

    def test_prediction_exp13_rf_regressor(self):
        """Test Random Forest Regressor (Glucose)"""
        res = self.client.post("/api/predict/exp13", json={
            "inputs": {
                "Pregnancies": 2, "BloodPressure": 75, "SkinThickness": 22,
                "Insulin": 110, "BMI": 28.5, "DiabetesPedigreeFunction": 0.40, "Age": 32
            }
        })
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Glucose", data["label"])

    def test_report_generation(self):
        """Test PDF report generation download route"""
        res = self.client.get("/report/exp02/download")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "application/pdf")

    def test_student_profile_and_navbar_layout(self):
        """Test profile picture, college logo, updated GitHub/LinkedIn links, repo link, and nav dock order"""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Profile image and college logo presence
        self.assertIn("/static/images/profile.jpg", html)
        self.assertIn("/static/images/college_logo.png", html)

        # Verified GitHub, Repo and LinkedIn links
        self.assertIn("https://github.com/Sudiksha29", html)
        self.assertIn("https://github.com/Sudiksha29/ML_ALL_Experiments", html)
        self.assertIn("https://www.linkedin.com/in/sudiksha-ingale-3bb37232b", html)

        # Learning center also has repo link
        res_learn = self.client.get("/learning-center")
        self.assertEqual(res_learn.status_code, 200)
        self.assertIn("https://github.com/Sudiksha29/ML_ALL_Experiments", res_learn.get_data(as_text=True))

        # No AI scholar text
        self.assertNotIn("AIML SCHOLAR", html)
        self.assertNotIn("AIML Scholar", html)
        self.assertNotIn("AI Scholar", html)

        # Nav dock order: Home 1st, then Experiments (13), Playground, Viva Guide
        dock_start = html.find('class="nav-links-dock"')
        self.assertNotEqual(dock_start, -1)
        dock_end = html.find("</div>", dock_start)
        dock_html = html[dock_start:dock_end]

        home_pos = dock_html.find("Home")
        exp_pos = dock_html.find("Experiments")
        playground_pos = dock_html.find("Playground")
        viva_pos = dock_html.find("Viva Guide")

        self.assertTrue(home_pos != -1 and exp_pos != -1 and playground_pos != -1 and viva_pos != -1)
        self.assertTrue(home_pos < exp_pos < playground_pos < viva_pos, "Expected Home, Experiments, Playground, Viva Guide order")

        # Static file serving checks
        res_img = self.client.get("/static/images/profile.jpg")
        self.assertEqual(res_img.status_code, 200)
        res_logo = self.client.get("/static/images/college_logo.png")
        self.assertEqual(res_logo.status_code, 200)

    def test_all_experiments_render_and_chart_handlers_exist(self):
        """Test all 13 experiment endpoints load cleanly with charts and parameter execution workspace"""
        for exp_id in [f"exp{i:02d}" for i in range(1, 14)]:
            res = self.client.get(f"/experiment/{exp_id}")
            self.assertEqual(res.status_code, 200, f"Failed for {exp_id}")
            html = res.get_data(as_text=True)
            self.assertIn("Select Practical Algorithm to Perform Experiment", html)
            if exp_id != "exp01":
                self.assertIn("expPredictionForm", html, f"Prediction form missing for {exp_id}")
                self.assertIn("Interactive Inference Workspace", html, f"Workspace missing for {exp_id}")
                self.assertIn("experimentCanvas", html, f"Canvas missing for {exp_id}")
                self.assertIn("MLCharts", html, f"MLCharts missing for {exp_id}")

    def test_svm_decision_boundaries_and_chart_renderers(self):
        """Test Linear SVM (exp10) and Non-Linear RBF SVM (exp11) have exact visual decision boundaries"""
        from services.model_registry import registry
        m10 = registry.get_metrics("exp10")
        self.assertIn("decision_boundary", m10)
        db10 = m10["decision_boundary"]
        self.assertIn("boundary_line", db10)
        self.assertIn("margin_plus", db10)
        self.assertIn("margin_minus", db10)
        self.assertIn("class0_samples", db10)
        self.assertIn("class1_samples", db10)
        self.assertIn("support_vectors", db10)
        self.assertGreater(len(db10["boundary_line"]), 10)
        self.assertGreater(len(db10["support_vectors"]), 5)

        m11 = registry.get_metrics("exp11")
        self.assertIn("decision_boundary", m11)
        db11 = m11["decision_boundary"]
        self.assertIn("boundary_curve", db11)
        self.assertIn("class0_samples", db11)
        self.assertIn("class1_samples", db11)
        self.assertIn("support_vectors", db11)
        self.assertGreater(len(db11["boundary_curve"]), 10)
        self.assertGreater(len(db11["support_vectors"]), 5)

        # Check frontend charts.js
        with open("static/js/charts.js", "r", encoding="utf-8") as f:
            charts_code = f.read()
        self.assertIn("renderLinearSVMDecisionBoundary", charts_code)
        self.assertIn("renderRBFDecisionBoundary", charts_code)

        # Check exp10 page has title and calls renderLinearSVMDecisionBoundary
        res10 = self.client.get("/experiment/exp10")
        self.assertEqual(res10.status_code, 200)
        html10 = res10.get_data(as_text=True)
        self.assertIn("Linear SVM Decision Boundary", html10)
        self.assertIn("renderLinearSVMDecisionBoundary", html10)

        # Check exp11 page has title and calls renderRBFDecisionBoundary
        res11 = self.client.get("/experiment/exp11")
        self.assertEqual(res11.status_code, 200)
        html11 = res11.get_data(as_text=True)
        self.assertIn("Non-Linear RBF SVM Curved Decision Boundary", html11)
        self.assertIn("renderRBFDecisionBoundary", html11)
        self.assertIn("resultFeaturesBox", html11)

    def test_prediction_returns_features_and_updates_chart(self):
        """Test that predictions return evaluated inputs and template contains feature display box"""
        res = self.client.post("/api/predict/exp11", json={
            "inputs": {"variance": -1.45, "skewness": 2.44, "curtosis": 1.88, "entropy": -0.76}
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("inputs", data)
        self.assertEqual(float(data["inputs"]["variance"]), -1.45)
        self.assertEqual(float(data["inputs"]["skewness"]), 2.44)

        # Check frontend charts.js handles userPoint
        with open("static/js/charts.js", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("userPoint", code)
        self.assertIn("Current Input", code)

if __name__ == "__main__":
    unittest.main()
