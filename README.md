# NE-EpiGuard — Waterborne Disease Prediction System
### Northeast India Public Health AI Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.18.0-orange)](https://mlflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

NE-EpiGuard is an end-to-end machine learning system for predicting waterborne disease outbreaks across Northeast India. It combines a stacking ensemble classifier with an AI-powered health advisory system (LangChain + Gemini) to provide disease risk assessment, WHO-compliant health guidance, and specialist doctor recommendations — all from a simple Streamlit interface.

The system supports both manual input and automated PDF report extraction, making it accessible to field health workers, public health researchers, and non-technical users across the 8 Northeast Indian states.

---

## Problem Statement

Northeast India contributes **35.8% of all zoonotic disease outbreaks** in India — the highest of any region. Districts in Assam, Meghalaya, Manipur, and Mizoram face recurring outbreaks of Cholera, Typhoid, Dysentery, Hepatitis A/E, Leptospirosis, and Giardiasis — driven by:

- Extreme monsoon seasonality and flooding
- Widespread untreated water consumption (55% of population)
- Poor sanitation infrastructure in remote hill districts
- Limited early warning systems

NE-EpiGuard addresses this gap by providing an AI-driven early detection and advisory tool grounded in real epidemiological data.

---

## Key Features

- **8-Class Disease Classification** — Cholera, Typhoid, Dysentery, Hepatitis A, Hepatitis E, Leptospirosis, Giardiasis, No Disease
- **Stacking Ensemble Model** — XGBoost + LightGBM + CatBoost → Logistic Regression meta-learner (Macro F1: **0.9212**)
- **MLflow Experiment Tracking** — 9 model runs tracked across baseline, tuning, and ensemble stages
- **PDF Report Upload** — Upload a health report PDF and Gemini auto-extracts all 33 fields
- **WHO-Compliant Advisory** — Structured health guidance following WHO WASH and emergency protocols
- **Doctor Recommendation** — Specialist and state helpline numbers triggered when prediction confidence ≥ 70%
- **Streamlit Deployment** — 4-tab interface: Manual Input → PDF Upload → Results → AI Advisory

---

## Disease Classes

| Disease | Transmission | Key Risk Factors |
|---|---|---|
| Cholera | Fecal-oral via water | Flooding, untreated river water, high fecal coliform |
| Typhoid | Contaminated water/food | Poor sanitation, high TDS, open defecation |
| Dysentery | Contaminated water | High turbidity, untreated sources |
| Hepatitis A | Fecal-oral | Poor handwashing, low WQI |
| Hepatitis E | Contaminated water | Flooding, low dissolved oxygen |
| Leptospirosis | Soil/flood water contact | Flooding, monsoon, animal contact |
| Giardiasis | Contaminated water | Stagnant pond/well water |
| No Disease | — | Good water quality + sanitation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Models | XGBoost 2.1.3, LightGBM 4.5.0, CatBoost 1.2.7, Scikit-learn 1.5.2 |
| Experiment Tracking | MLflow 2.18.0 |
| LLM Integration | LangChain 0.2.16 + Google Gemini 3.6 Flash |
| PDF Extraction | pdfplumber 0.11.4 |
| Frontend | Streamlit 1.41.1 |
| Serialization | dill 0.3.9 |
| Data | Pandas 2.2.3, NumPy 1.26.4 |

---

## Project Structure

```
NE-EpiGuard/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Raw data loading + train/test split
│   │   ├── data_transformation.py   # Preprocessing pipeline
│   │   ├── feature_engineering.py   # Custom sklearn transformer
│   │   └── model_trainer.py         # Stacking ensemble + MLflow
│   ├── pipeline/
│   │   ├── train_pipeline.py        # End-to-end training pipeline
│   │   └── predict_pipeline.py      # Prediction + doctor recommendation
│   ├── langchain_helper.py          # PDF extraction + WHO advisory
│   ├── utils.py                     # Encodings, mappings, save/load
│   ├── logger.py                    # Logging setup
│   └── exception.py                 # Custom exception handler
├── Notebooks/
│   ├── EDA.ipynb                    # Exploratory Data Analysis (15 sections)
│   ├── Model_Training.ipynb         # Baseline + tuning + stacking
│   └── Data/
│       └── northeast_waterborne.csv # 405,074 records, 39 features
├── artifacts/
│   ├── model.pkl                    # Trained stacking ensemble
│   ├── preprocessor.pkl             # Fitted StandardScaler pipeline
│   └── le_target.pkl                # Label encoder for disease classes
├── sample_reports/
│   ├── sample_report_cholera.pdf    # Test PDF — Cholera patient
│   ├── sample_report_typhoid.pdf    # Test PDF — Typhoid patient
│   └── sample_report_healthy.pdf   # Test PDF — Healthy person
├── mlflow/                          # MLflow experiment runs
├── app.py                           # Streamlit application
├── requirements.txt
├── setup.py
└── .python-version                  # Python 3.12
```

---

## Dataset

**404,074 patient records** across 8 Northeast Indian states (Assam, Manipur, Meghalaya, Mizoram, Nagaland, Arunachal Pradesh, Sikkim, Tripura) with 39 features:

- **Demographic**: age, gender, state, district
- **Water Quality**: pH, turbidity, dissolved oxygen, BOD, fecal coliform, total coliform, TDS, nitrate, fluoride, arsenic, WQI
- **Sanitation**: water source, water treatment, handwashing practice, toilet access, open defecation rate, sewage treatment %
- **Climate**: temperature, rainfall, humidity, flooding, season, month
- **Symptoms**: 8 binary flags (diarrhea, vomiting, fever, abdominal pain, dehydration, jaundice, bloody stool, skin rash)
- **Target**: disease (8 classes)

---

## Model Performance

| Model | Macro F1 | Accuracy | Lepto Recall | Cholera Recall |
|---|---|---|---|---|
| Logistic Regression | 0.8243 | 0.8633 | 0.7494 | 0.9484 |
| Random Forest | 0.9091 | 0.9302 | 0.7736 | 0.9811 |
| Extra Trees | 0.8534 | 0.8856 | 0.6902 | 0.9576 |
| XGBoost (tuned) | 0.9201 | — | — | — |
| LightGBM (tuned) | 0.9201 | — | — | — |
| CatBoost (tuned) | 0.9156 | — | — | — |
| **Stacking Ensemble** | **0.9212** | **0.9398** | **0.8547** | **0.9820** |

**Primary metric: Macro F1** — chosen to handle class imbalance (3.98x ratio) and ensure Leptospirosis (clinically most critical, 7% of cases) is weighted equally with majority classes.

---

## Feature Engineering

Four engineered features created from domain knowledge and EDA insights:

| Feature | Formula | Insight |
|---|---|---|
| `symptom_count` | Sum of 8 binary symptom flags | Severity proxy — Dysentery avg 4.2, No_Disease ~0.1 |
| `source_treatment_risk` | `(6 - water_source) + (3 - water_treatment)` | River + Untreated = highest risk (9), Piped + Chlorinated = 0 |
| `age_hygiene_risk` | `age × (2 - handwashing_encoded)` | Children who never wash hands = highest vulnerability |
| `contamination_index` | `log(fecal_coliform) + log(total_coliform) + turbidity/10` | Composite contamination severity |

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/sumeet-016/disease.git
cd disease
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Add API key

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free API key at [aistudio.google.com](https://aistudio.google.com)

### 3. Train the model (first time only)

```bash
python src/pipeline/train_pipeline.py
```

This generates `artifacts/model.pkl`, `artifacts/preprocessor.pkl`, and `artifacts/le_target.pkl`.

### 4. Run the app

```bash
streamlit run app.py
```

---

## How to Use

### Option A — Manual Input
Fill in patient details, water quality parameters, and symptoms in Tab 1 → click Predict.

### Option B — PDF Upload
Upload one of the sample PDFs from `sample_reports/` or your own structured health report → Gemini auto-extracts all fields → click Predict.

### Sample PDFs
Three test reports are provided in `sample_reports/`:
- `sample_report_cholera.pdf` — Expected: Cholera (high confidence)
- `sample_report_typhoid.pdf` — Expected: Typhoid (high confidence)
- `sample_report_healthy.pdf` — Expected: No Disease

---

## LangChain Advisory System

When a disease is predicted, the LangChain + Gemini pipeline generates a WHO-compliant health advisory covering:

1. **Disease Overview** — WHO definition and South Asia burden
2. **Immediate Actions** — WHO Emergency Protocol including ORS
3. **Water Safety Tips** — WHO WASH guidelines for the patient's specific water source
4. **Prevention** — WHO 5 Moments of Hand Hygiene and sanitation measures
5. **Warning Signs** — WHO Red Flag symptoms requiring hospitalization
6. **Community Action** — WHO One Health approach for outbreak prevention
7. **Helpline Numbers** *(triggered when confidence ≥ 70%)* — State-specific disease control officer, 104, 108, NCDC

---

## EDA Key Findings

- **Monsoon drives 5x disease surge** — Cholera peaks at 23,000 cases in June vs 4,000 in January
- **Fecal coliform is the strongest predictor** — No_Disease ~50 CFU/100ml vs Cholera ~3,000 CFU
- **Untreated water = 15% Cholera rate** vs Chlorinated water = 3%
- **Always handwashing = 93% No_Disease** vs Never handwashing = 18% No_Disease
- **Jaundice alone discriminates Hepatitis** (80-85% prevalence) from all other diseases
- **Skin rash alone discriminates Leptospirosis** (64.7%) from all other diseases

---

## MLflow Experiment Tracking

All 9 model runs are tracked in MLflow with parameters, metrics, and confusion matrix artifacts.

To view the MLflow UI locally:
```bash
mlflow ui --backend-store-uri file:///path/to/NE-EpiGuard/mlflow
```
Then open `http://localhost:5000`

---

## States Covered

Assam · Manipur · Meghalaya · Mizoram · Nagaland · Arunachal Pradesh · Sikkim · Tripura

---

## Author

**Sumeet Kumar Pal**
B.Tech CS & Data Science | Oriental Institute of Science & Technology, Bhopal
GitHub: [sumeet-016](https://github.com/sumeet-016)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
