/**
 * ML LabVerse — Interactive Visualizations Service (Chart.js wrapper)
 */

const MLCharts = {
    // Helper to clear existing chart on canvas
    _prepareCanvas: (canvasId) => {
        const existing = Chart.getChart(canvasId);
        if (existing) {
            existing.destroy();
        }
        return document.getElementById(canvasId)?.getContext("2d");
    },

    // 1. Scatter Plot with Best Fit Line (Simple Linear Regression)
    renderScatterLine: (canvasId, scatterData, lineData, xLabel = "Feature (X)", yLabel = "Target (Y)", userPoint = null) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const datasets = [
            {
                label: "Training Samples",
                data: scatterData,
                backgroundColor: "rgba(57, 125, 120, 0.6)",
                borderColor: "#397D78",
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: "Fitted Regression Line (y = mx + c)",
                data: lineData,
                type: "line",
                borderColor: "#C05621",
                backgroundColor: "transparent",
                borderWidth: 2.5,
                pointRadius: 0,
                fill: false
            }
        ];

        if (userPoint && userPoint.x !== undefined && userPoint.y !== undefined) {
            datasets.push({
                label: `★ Current Input (${userPoint.x}, ${userPoint.y}): ${userPoint.label || 'Predicted'}`,
                data: [{ x: userPoint.x, y: userPoint.y }],
                type: "scatter",
                backgroundColor: "#F59E0B",
                borderColor: "#0F172A",
                borderWidth: 3,
                pointRadius: 11,
                pointHoverRadius: 14,
                pointStyle: "star"
            });
        }

        return new Chart(ctx, {
            type: "scatter",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top" },
                    tooltip: { mode: "nearest" }
                },
                scales: {
                    x: { title: { display: true, text: xLabel } },
                    y: { title: { display: true, text: yLabel } }
                }
            }
        });
    },

    // 2. Actual vs Predicted Bar Chart
    renderActualVsPredicted: (canvasId, actuals, predicteds) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const labels = actuals.map((_, i) => `Test #${i+1}`);

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Actual Value",
                        data: actuals,
                        backgroundColor: "rgba(57, 125, 120, 0.8)",
                        borderRadius: 4
                    },
                    {
                        label: "Predicted Value",
                        data: predicteds,
                        backgroundColor: "rgba(237, 235, 250, 0.9)",
                        borderColor: "#5B4E9A",
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "top" } },
                scales: {
                    y: { beginAtZero: false, title: { display: true, text: "Output Value" } }
                }
            }
        });
    },

    // 3. ROC Curve
    renderROCCurve: (canvasId, fprs, tprs, aucVal) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const curveData = fprs.map((fpr, i) => ({ x: fpr, y: tprs[i] }));
        const randomLine = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

        return new Chart(ctx, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: `Model ROC (AUC = ${aucVal.toFixed(3)})`,
                        data: curveData,
                        showLine: true,
                        borderColor: "#397D78",
                        backgroundColor: "rgba(57, 125, 120, 0.1)",
                        fill: true,
                        borderWidth: 2.5,
                        pointRadius: 2
                    },
                    {
                        label: "Random Guess Baseline",
                        data: randomLine,
                        showLine: true,
                        borderColor: "#A0AEC0",
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { min: 0, max: 1, title: { display: true, text: "False Positive Rate (1 - Specificity)" } },
                    y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate (Sensitivity)" } }
                }
            }
        });
    },

    // 4. Feature Importance Horizontal Bar Chart
    renderFeatureImportance: (canvasId, featureData) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const labels = featureData.map(d => d.feature);
        const values = featureData.map(d => d.importance);

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Relative Feature Importance",
                    data: values,
                    backgroundColor: "rgba(57, 125, 120, 0.75)",
                    borderColor: "#397D78",
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (c) => `Importance: ${(c.raw * 100).toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: { beginAtZero: true, title: { display: true, text: "Gini Importance Score" } }
                }
            }
        });
    },

    // 5. K-Tuning Curve (KNN)
    renderKTuning: (canvasId, kValues, cvScores, bestK) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const pointBackgrounds = kValues.map(k => k === bestK ? "#C05621" : "#397D78");
        const pointRadii = kValues.map(k => k === bestK ? 7 : 4);

        return new Chart(ctx, {
            type: "line",
            data: {
                labels: kValues,
                datasets: [{
                    label: "10-Fold CV Accuracy",
                    data: cvScores,
                    borderColor: "#397D78",
                    backgroundColor: "rgba(57, 125, 120, 0.08)",
                    borderWidth: 2,
                    pointBackgroundColor: pointBackgrounds,
                    pointRadius: pointRadii,
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "Number of Neighbors (K)" } },
                    y: { title: { display: true, text: "Mean Cross-Validation Accuracy" } }
                }
            }
        });
    },

    // 6. Gaussian Class Means Comparison (Gaussian Naive Bayes - EXP 08)
    renderGaussianClassMeans: (canvasId, features, stats0, stats1, classLabels = ["Counterfeit (0)", "Genuine (1)"]) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const labels = features.map(f => f.charAt(0).toUpperCase() + f.slice(1));
        const means0 = features.map(f => stats0 && stats0[f] ? stats0[f].mean : 0);
        const means1 = features.map(f => stats1 && stats1[f] ? stats1[f].mean : 0);

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: `${classLabels[0]} Mean (μ₀)`,
                        data: means0,
                        backgroundColor: "rgba(225, 29, 72, 0.75)",
                        borderColor: "#BE123C",
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: `${classLabels[1]} Mean (μ₁)`,
                        data: means1,
                        backgroundColor: "rgba(16, 185, 129, 0.75)",
                        borderColor: "#059669",
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.dataset.label}: ${c.raw.toFixed(3)}`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: "Continuous Feature Dimension" } },
                    y: { title: { display: true, text: "Learned Normal Mean Parameter (μ)" } }
                }
            }
        });
    },

    // 7. Multiclass Classification Breakdown (Multiclass Logistic Regression - EXP 05)
    renderMulticlassBreakdown: (canvasId, classes, correctCounts, totalCounts) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: classes,
                datasets: [
                    {
                        label: "Correctly Classified Samples",
                        data: correctCounts,
                        backgroundColor: "rgba(104, 91, 199, 0.8)",
                        borderColor: "#5B4E9A",
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: "Total Held-Out Test Samples",
                        data: totalCounts,
                        backgroundColor: "rgba(226, 232, 240, 0.8)",
                        borderColor: "#94A3B8",
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top" }
                },
                scales: {
                    x: { title: { display: true, text: "Target Classes (Softmax Cross-Entropy)" } },
                    y: { beginAtZero: true, title: { display: true, text: "Sample Count" } }
                }
            }
        });
    },

    // 8. Linear SVM Decision Boundary & Margins (EXP 10)
    renderLinearSVMDecisionBoundary: (canvasId, dbData, userPoint = null) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const c0 = dbData.class0_samples || [];
        const c1 = dbData.class1_samples || [];
        const sv = dbData.support_vectors || [];
        const line = dbData.boundary_line || [];
        const mPlus = dbData.margin_plus || [];
        const mMinus = dbData.margin_minus || [];

        const datasets = [
            {
                label: "Class 0: Counterfeit",
                data: c0,
                backgroundColor: "rgba(225, 29, 72, 0.7)",
                borderColor: "#BE123C",
                pointRadius: 4.5,
                pointHoverRadius: 6.5
            },
            {
                label: "Class 1: Genuine",
                data: c1,
                backgroundColor: "rgba(16, 185, 129, 0.75)",
                borderColor: "#059669",
                pointRadius: 4.5,
                pointHoverRadius: 6.5
            },
            {
                label: "Margin Support Vectors (SV)",
                data: sv,
                backgroundColor: "rgba(245, 158, 11, 0.9)",
                borderColor: "#D97706",
                pointRadius: 7.5,
                pointHoverRadius: 9,
                pointStyle: "rectRot"
            },
            {
                label: "Optimal Hyperplane (w·x + b = 0)",
                data: line,
                type: "line",
                borderColor: "#4338CA",
                borderWidth: 2.5,
                pointRadius: 0,
                fill: false
            },
            {
                label: "Positive Margin (+1)",
                data: mPlus,
                type: "line",
                borderColor: "rgba(67, 56, 202, 0.55)",
                borderWidth: 1.5,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            },
            {
                label: "Negative Margin (-1)",
                data: mMinus,
                type: "line",
                borderColor: "rgba(67, 56, 202, 0.55)",
                borderWidth: 1.5,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }
        ];

        if (userPoint && userPoint.x !== undefined && userPoint.y !== undefined) {
            const isGenuine = userPoint.prediction === 1 || String(userPoint.classLabel || "").includes("Genuine");
            datasets.push({
                label: `★ Current Input (${userPoint.x}, ${userPoint.y}): ${userPoint.classLabel || (isGenuine ? 'Genuine Banknote (1)' : 'Counterfeit Banknote (0)')}`,
                data: [{ x: userPoint.x, y: userPoint.y }],
                backgroundColor: isGenuine ? "#10B981" : "#E11D48",
                borderColor: "#0F172A",
                borderWidth: 3,
                pointRadius: 11,
                pointHoverRadius: 14,
                pointStyle: "star"
            });
        }

        return new Chart(ctx, {
            type: "scatter",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.dataset.label}: (${c.raw.x}, ${c.raw.y})`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: "Wavelet Variance (X₁)" } },
                    y: { title: { display: true, text: "Wavelet Skewness (X₂)" } }
                }
            }
        });
    },

    // 9. Non-Linear RBF SVM Curved Decision Boundary (EXP 11)
    renderRBFDecisionBoundary: (canvasId, dbData, userPoint = null) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const c0 = dbData.class0_samples || [];
        const c1 = dbData.class1_samples || [];
        const sv = dbData.support_vectors || [];
        const curve = dbData.boundary_curve || [];

        const datasets = [
            {
                label: "Class 0: Counterfeit",
                data: c0,
                backgroundColor: "rgba(225, 29, 72, 0.7)",
                borderColor: "#BE123C",
                pointRadius: 4.5,
                pointHoverRadius: 6.5
            },
            {
                label: "Class 1: Genuine",
                data: c1,
                backgroundColor: "rgba(16, 185, 129, 0.75)",
                borderColor: "#059669",
                pointRadius: 4.5,
                pointHoverRadius: 6.5
            },
            {
                label: "RBF Support Vectors (Kernel Anchors)",
                data: sv,
                backgroundColor: "rgba(245, 158, 11, 0.9)",
                borderColor: "#D97706",
                pointRadius: 7.5,
                pointHoverRadius: 9,
                pointStyle: "triangle"
            },
            {
                label: "Non-Linear RBF Decision Boundary",
                data: curve,
                type: "line",
                borderColor: "#7C3AED",
                borderWidth: 2.8,
                pointRadius: 0,
                tension: 0.3,
                fill: false
            }
        ];

        if (userPoint && userPoint.x !== undefined && userPoint.y !== undefined) {
            const isGenuine = userPoint.prediction === 1 || String(userPoint.classLabel || "").includes("Genuine");
            datasets.push({
                label: `★ Current Input (${userPoint.x}, ${userPoint.y}): ${userPoint.classLabel || (isGenuine ? 'Genuine Banknote (1)' : 'Counterfeit Banknote (0)')}`,
                data: [{ x: userPoint.x, y: userPoint.y }],
                backgroundColor: isGenuine ? "#10B981" : "#E11D48",
                borderColor: "#0F172A",
                borderWidth: 3,
                pointRadius: 11,
                pointHoverRadius: 14,
                pointStyle: "star"
            });
        }

        return new Chart(ctx, {
            type: "scatter",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.dataset.label}: (${c.raw.x}, ${c.raw.y})`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: "Wavelet Variance (X₁)" } },
                    y: { title: { display: true, text: "Wavelet Skewness (X₂)" } }
                }
            }
        });
    },

    // 10. Linear SVM Margin & Support Vector Breakdown (Legacy bar chart helper)
    renderSVMBreakdown: (canvasId, svCount, totalSamples, class0Correct, class1Correct) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Class 0 Correct", "Class 1 Correct", "Margin Support Vectors"],
                datasets: [{
                    label: "Sample Count",
                    data: [class0Correct, class1Correct, svCount],
                    backgroundColor: [
                        "rgba(225, 29, 72, 0.75)",
                        "rgba(16, 185, 129, 0.75)",
                        "rgba(245, 158, 11, 0.8)"
                    ],
                    borderColor: [
                        "#BE123C",
                        "#059669",
                        "#D97706"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.label}: ${c.raw} samples`
                        }
                    }
                },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: "Sample Count" } }
                }
            }
        });
    },

    // 11. Kernel Comparison: Linear vs Non-Linear RBF (Legacy bar chart helper)
    renderKernelComparison: (canvasId, compData) => {
        const ctx = MLCharts._prepareCanvas(canvasId);
        if (!ctx) return;

        const labels = ["Linear Kernel SVC", "RBF Kernel SVC (Non-Linear)"];
        const accuracies = [
            (compData.linear_accuracy * 100).toFixed(1),
            (compData.rbf_accuracy * 100).toFixed(1)
        ];
        const svCounts = [compData.linear_sv_count, compData.rbf_sv_count];

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Test Accuracy (%)",
                        data: accuracies,
                        backgroundColor: "rgba(16, 185, 129, 0.75)",
                        borderColor: "#059669",
                        borderWidth: 1.5,
                        borderRadius: 6,
                        yAxisID: "y"
                    },
                    {
                        label: "Support Vectors Count",
                        data: svCounts,
                        backgroundColor: "rgba(104, 91, 199, 0.8)",
                        borderColor: "#5B4E9A",
                        borderWidth: 1.5,
                        borderRadius: 6,
                        yAxisID: "y1"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top" }
                },
                scales: {
                    y: {
                        type: "linear",
                        display: true,
                        position: "left",
                        min: 90,
                        max: 100,
                        title: { display: true, text: "Accuracy (%)" }
                    },
                    y1: {
                        type: "linear",
                        display: true,
                        position: "right",
                        grid: { drawOnChartArea: false },
                        title: { display: true, text: "Support Vectors Count" }
                    }
                }
            }
        });
    }
};
