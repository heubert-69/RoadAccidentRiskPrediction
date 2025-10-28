# 🚦 Accident Risk Prediction — Robust ML Pipeline for Road Safety Kaggle Competition

> **Model:** CatBoost Regressor (RMSE = 0.05563)  
> **Focus:** Feature Engineering • Model Robustness • MLOps Pipeline • Explainability  

---

## 📘 Overview

This project builds a **production-ready machine learning pipeline** to predict the likelihood of **road accident risk** given road, weather, and environmental conditions.  

It integrates:
- **Advanced feature engineering**
- **Gradient boosting ensemble models (CatBoost, LightGBM, XGBoost)**
- **Robustness testing** under noise, missing data, and domain shift
- **Explainability (SHAP)**
- **Automated retraining and monitoring (MLOps-ready design)**

The model achieves **0.05563 RMSE** on the official test dataset — just **0.00027 behind the global leaderboard’s top score (0.05536)**, placing it in **elite-performance range**.

---

## 🧠 Core Problem

Given structured tabular data containing:
- Road geometry & curvature  
- Environmental and traffic conditions  
- Time-of-day & seasonality indicators  
- Accident frequency statistics  

Predict the **`accident_risk`** for each observation — a continuous metric representing the likelihood or severity of traffic accidents.

---

## 🧩 Key Features

### 🧮 1. Feature Engineering
Implemented in [`feature_engineer.py`](./feature_engineer.py):

| Feature | Description |
|----------|--------------|
| `num_reported_accidents_log` | Log transform for skew correction |
| `traffic_density` | Ratio of lanes to speed limit |
| `curvature_intenisty` | Combined curvature–speed risk factor |
| `risk_exposure` | Exposure measure per lane |
| `congestion_risk` | Binary flag for holiday + school season interaction |
| `mean_accidents` | Group mean by road type |
| `mean_accidents_deviation` | Local deviation from mean accidents |

These derived features capture **interaction and contextual risk patterns** that raw features miss.

---

### 🧰 2. Model Stack

| Model | RMSE | R² | Notes |
|--------|------|----|------|
| CatBoost | **0.05563** | 0.8854 | Primary model |
| LightGBM | 0.0563 | 0.8854 | Secondary ensemble |
| XGBoost | 0.0565 | 0.8842 | Third ensemble |

The final model uses **CatBoost** for its:
- Superior categorical handling  
- Regularization through ordered boosting  
- Built-in missing value tolerance  
- SHAP interpretability

---

### 🧪 3. Robustness Testing

| Stress Type | Δ (MAE diff) | Interpretation |
|--------------|---------------|----------------|
| Gaussian Noise (10%) | **0.0080** | Excellent noise stability |
| Missing Data | **0.0255** | Moderate degradation, robust handling |
| Domain Shift | Varies by feature | Acceptable generalization margin |

**Conclusion:**  
Model is **production-safe**, resilient to sensor noise, missing inputs, and moderate domain drift.

---

### 📊 4. Explainability (SHAP)

- Feature importance derived from **SHAP values**
- Insights reveal `curvature_intenisty`, `traffic_density`, and `risk_exposure` as the dominant predictors
- Used for model debugging, fairness checks, and stakeholder reporting

---

### ⚙️ 5. MLOps Design (Extendable)

The pipeline was built with MLOps readiness in mind:
- `feature_engineer.py` modularized for reuse  
- Easily integrated with **MLflow** or **Airflow** for automated retraining  
- Scalable deployment via **FastAPI + Docker + AWS SageMaker**  
- Supports model registry and version control

---

## 🧩 Workflow
```mermaid
flowchart TD
A["Data Ingestion"] --> B["Feature Engineering (feature_engineer.py)"]
B --> C["Label Encoding + Imputation"]
C --> D["Model Training (CatBoost)"]
D --> E["Cross-Validation + Statistical Tests"]
E --> F["Robustness Stress Tests"]
F --> G["Explainability (SHAP)"]
G --> H["Prediction + Submission CSV"]


🧪 Statistical Validation
Paired tests on cross-validation folds confirmed statistically significant model differences:

Comparison	t-test p	Wilcoxon p	Conclusion
CatBoost vs LGBM	0.015	0.002	CatBoost > LGBM
CatBoost vs XGB	1.86e-32	1.15e-24	CatBoost ≫ XGB
LGBM vs XGB	1.93e-42	2.64e-32	LGBM > XGB

This ensures performance improvements are statistically reliable.

🚀 Kaggle Submission
Final output format:

python-repl
Copy code
id,accident_risk
1,0.318
2,0.442
3,0.276
...
Generated via:

```python
submission.to_csv("submission.csv", index=False)
```

🧰 Tech Stack
- Layer	Tools
- ML Framework	CatBoost, LightGBM, XGBoost
- Preprocessing	scikit-learn, NumPy, pandas
- Visualization	Matplotlib, Seaborn, SHAP
- Statistical Testing	SciPy
- MLOps (ready)	MLflow, Docker, FastAPI, AWS SageMaker
- Environment	Python 3.12, Jupyter Notebook

📦 Repository Structure
```bash
📂 RoadAccidentRiskPrediction/
│
├── feature_engineer.py           # Feature transformation logic
├── train_pipeline.ipynb          # Model training, validation, SHAP
├── stress_testing.ipynb          # Noise, missing data, and shift tests
├── submission_pipeline.py        # Final test set prediction
├── models/
│   └── catboost_model.cbm        # Saved model
├── data/
│   ├── train.csv
│   └── test.csv
└── README.md                     # This file
```

📈 Results Summary
Metric	Train CV	Test	Δ
RMSE	0.0563	0.05563	↓ 0.0007
R²	0.8865	0.8854	Stable
Leaderboard Δ	—	+0.00027	Within 0.5% of top global score

🧭 Future Improvements
- Implement rank-based ensemble (CatBoost + LGBM + XGB)

- Add automated data drift detection and retraining trigger

- Deploy via FastAPI + Docker container to AWS SageMaker

- Extend to causal modeling for interpretability in policy planning

