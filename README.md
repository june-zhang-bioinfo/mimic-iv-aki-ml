# 🏥 AKI Phenotype Discovery and Outcome Prediction in MIMIC-IV

A comprehensive machine learning pipeline for discovery distinct acute kidney injury (AKI) phenotypes and predicting clinical outcomes using the MIMIC-IV database.

## 📋 Overview

This project implements two machine learning approaches to understanding AKI in critically ill patients:

1. **Unsupervised Clustering**: Discovers distinct AKI phenotypes based on clinical features from the 24 hours following AKI onset.
2. **Supervised Prediction**: Predicts severe outcomes using features from the 24 hours prior to AKI onset.

---

## 📊 Dataset

- **Source**: MIMIC-IV v3.1
- **AKI Definition**: KDIGO creatinine criteria
  - Criterion 1: ≥0.3 mg/dL increase within 48 hours
  - Criterion 2: ≥1.5× baseline within 7 days
- **Inclusion Criteria**: 
  - ICU stay ≥48 hours
  - Valid baseline creatinine (first measurement within 24h of ICU admission)
- **Feature Categories**: 
  - **Demographics**: age, gender, race, admission type, ICU type
  - **Comorbidities**: CHF, hypertension, diabetes, CKD, liver disease, COPD, malignancy
  - **Vital Signs**: heart rate, mean arterial pressure (MAP), respiratory rate, temperature
  - **Laboratory Values**: creatinine, BUN, potassium, bicarbonate, lactate, pH, hemoglobin, platelets, WBC, bilirubin
  - **Derived Features**: shock index (HR/MAP), BUN/Cr ratio, creatinine fold-change, comorbidity count, missingness indicators

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
umap-learn >= 0.5.0
shap >= 0.40.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
tqdm >= 4.62.0
joblib >= 1.1.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mimic-aki-phenotypes.git
cd mimic-aki-phenotypes

# Create virtual environment
python3 -m venv aki_env
source aki_env/bin/activate  # On Windows: aki_env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost umap-learn shap matplotlib seaborn scipy tqdm joblib
```

### Data Setup

1. **Obtain MIMIC-IV access**:
   - Complete CITI training: https://physionet.org/about/citi-course/
   - Request access: https://physionet.org/content/mimiciv/
   - Download MIMIC-IV v3.1

2. **Update configuration**:
   ```python
   # In each script (Untitled-1.py, Untitled-4.py), update:
   MIMIC_PATH = "/path/to/your/mimiciv/3.1"
   OUTPUT_DIR = "/path/to/output"
   ```

3. **Verify required tables**:
   ```
   mimiciv/3.1/
   ├── icu/
   │   ├── icustays.csv.gz
   │   └── chartevents.csv.gz
   └── hosp/
       ├── labevents.csv.gz
       ├── patients.csv.gz
       ├── admissions.csv.gz
       └── diagnoses_icd.csv.gz
   ```

---

## 🔬 Methodology

### Data Preprocessing 

#### Missing Data Handling
```
Imputation Strategy (by Missingness Level):
├── >50% missing → Drop feature
├── 30-50% missing → KNN imputation (k=5, distance-weighted)
├── 10-30% missing → Median imputation
└── <10% missing → Median imputation

Special Handling:
├── MAP measurements: Coalesce invasive/non-invasive (prioritize invasive)
├── Potassium duplicates: Drop high-missingness itemid (>70%)
└── Missingness indicators: Binary flags for key labs (has_lactate, has_abg, etc.)
```

#### Feature Engineering
- **AKI severity**: `cr_fold_change = aki_cr / baseline_cr`, `cr_change_pct`
- **Hemodynamics**: `shock_index = hr / map`, `hypotensive = (map < 65)`
- **Renal function**: `bun_cr_ratio = bun / cr`
- **Comorbidity burden**: `comorbidity_count = sum(chf, diabetes, ckd, ...)`
- **Age stratification**: Categorical bins [0-40, 40-60, 60-75, 75+]
- **Laboratory thresholds**: `thrombocytopenia = (platelets < 150)`

#### Scaling & Normalization
- **Method**: RobustScaler (median centering, IQR scaling)
- **Rationale**: Resistant to outliers common in ICU data
- **Application**: Continuous features only; binary features unscaled
- **Leakage prevention**: Fit on training set only, transform val/test

---

### Clustering Approach

#### Dimensionality Reduction
```
PCA:
├── Components: Up to 50 (or number of features, whichever is smaller)
├── Selection: Components explaining 90% variance (~30-40 typical)
└── Purpose: Curse of dimensionality mitigation, clustering input

UMAP:
└── Purpose: Visualization only (not used for clustering)
```


#### Algorithms 

1. **K-Means**
   - Init: k-means++ (default)
   - Runs: 50 (n_init=50)
   - Max iterations: 1000
   - Fast, assumes spherical clusters

2. **Hierarchical Clustering**
   - Linkage: Ward (minimizes within-cluster variance)
   - Agglomerative approach
   - Produces dendrogram for interpretability

3. **Gaussian Mixture Model (GMM)**
   - Covariance: Full (allows ellipsoidal clusters)
   - Soft clustering: Outputs probability per cluster
   - Runs: 20 (n_init=20)

4. **DBSCAN** (exploratory)
   - eps: Auto-selected via 95th percentile of k-distances
   - min_samples: 10
   - Identifies noise/outliers


#### Optimal K Selection
- **Range tested**: k = 2 to 8
- **Metrics evaluated**:
  - Elbow method (inertia/within-cluster sum of squares)
  - Silhouette score (maximize, range [-1, 1])
  - Davies-Bouldin index (minimize, lower is better)
  - Calinski-Harabasz score (maximize, higher is better)
- **Selection**: Majority voting across metrics OR manual selection based on clinical interpretability

---

### Prediction Pipeline

#### Train/Validation/Test Split
```
Total Cohort
├── 70% Training (fit models + hyperparameters)
├── 15% Validation (hyperparameter selection)
└── 15% Test (final evaluation, never seen during training)

Stratification: Primary outcome (severe_aki) to maintain class balance
Random seed: 42 (reproducibility)
```


#### Multi-Label Classification Strategy
- **Targets**: 4 binary outcomes (severe_aki, progression, mortality, prolonged_icu)
- **Wrapper**: `MultiOutputClassifier` for single-label base models
- **Class imbalance handling**:
  - Logistic Regression / Random Forest / MLP: `class_weight='balanced'`
  - XGBoost: `scale_pos_weight = neg_count / pos_count` (per outcome)


### Model Training & THyperparameter uning

We trained multiple baseline and non-linear models (logistic regression, random forest, XGBoost, and MLP).
Hyperparameters were optimized on a validation set using grid or random search, with **ROC-AUC as the primary
selection criterion**. XGBoost models were trained **separately for each outcome**.
Full hyperparameter search spaces and tuning strategies are documented in `docs/model_tuning.md`.

---

### Model Interpretation

#### Feature Importance
- **Tree-based models**: Gain-based importance (average impurity reduction)
- **Ranking**: Top 20 features by average importance
- **Validation**: Cross-check consistency between Random Forest and XGBoost
- **Output**: `output/xgb_feature_importance.csv`


#### SHAP (SHapley Additive exPlanations)
```
Analysis per outcome:
├── Explainer: TreeExplainer (XGBoost models)
├── Sample: 1,000 random test patients (computational efficiency)
├── Visualizations:
│   ├── Beeswarm plot: Feature directionality + magnitude
│   ├── Heatmap: Patient-level consistency
│   └── Gain bar chart: Feature importance ranking
└── Interpretation: Bidirectional effects revealed
    (e.g., high lactate → increased risk, low MAP → increased risk)

Output: output/interpretation/beeswarm_*.png, heatmap_*.png, gain_*.png
```

---


## 🎯 Key Results 

### Phenotype Discovery

* **Identified Phenotypes**: *2* major distinct AKI phenotypes
* **Clustering Quality**:
  - Silhouette Score: `[CHECK: output/clustering_metrics.csv - Silhouette Score]`  
  - Davies-Bouldin Index: `[CHECK: output/clustering_metrics.csv - Davies-Bouldin Index]`  
  - Calinski-Harabasz Score: `[CHECK: output/clustering_metrics.csv - Calinski-Harabasz Score]`
* **Top differentiators**:

---

### Outcome Prediction

* **Best Model**: Gradient-boosted trees (XGBoost)
* **Performance**: 
| Outcome | ROC-AUC | PR-AUC | F1 Score |
|---------|---------|--------|----------|
| **Severe AKI** | `[Test_ROC_AUC]` | `[Test_PR_AUC]` | `[Test_F1]` |
| **Progression** | `[Test_ROC_AUC]` | `[Test_PR_AUC]` | `[Test_F1]` |
| **Mortality** | `[Test_ROC_AUC]` | `[Test_PR_AUC]` | `[Test_F1]` |
| **Prolonged ICU** | `[Test_ROC_AUC]` | `[Test_PR_AUC]` | `[Test_F1]` |
* **SHAP Analysis**:

---

## 📊 Detailed Results

➡️ Full quantitative tables, per-outcome metrics, and statistical tests are provided in:

* `output/clustering_metrics.csv`
* `output/cluster_summary.csv`
* `output/cluster_heatmap.png`
* `output/model_comparison.csv`
* `output/*_roc_curves_val.png`

---

### Clustering Insights

**Beyond KDIGO Staging**:
- Phenotypes cut across severity grades → distinct pathophysiology beyond creatinine
- `[Cluster X]` has `[%]` Stage 1 but `[high/low]` mortality → severity ≠ outcome

**Actionable Stratification** (examples to customize based on your clusters):
- **Cluster with shock phenotype** → Early vasopressor optimization, consider early RRT
- **Cluster with multi-organ dysfunction** → Goals of care discussion, ICU resource planning
- **Cluster with stable hemodynamics** → Conservative fluid management, avoid nephrotoxins

**Research Applications**:
- **Clinical trial enrichment**: Homogeneous subgroups for targeted interventions
- **Comparative effectiveness**: Treatment heterogeneity by phenotype
- **Biomarker discovery**: Phenotype-specific molecular signatures

---

### Prediction Utility

**Early Warning System**:
- 24-hour lead time enables proactive intervention before AKI onset
- Risk scores can trigger automated alerts in EHR systems

**Clinical Decision Support** (suggested thresholds - customize based on your ROC/PR curves):
```
High Risk (Top 10%):
├── P(severe_aki) > [threshold] → Nephrology consult, intensify monitoring
├── P(mortality) > [threshold] → Goals of care discussion, family meeting
└── P(prolonged_icu) > [threshold] → Early mobilization, discharge planning

Medium Risk (Next 20%):
├── Enhanced surveillance
└── Avoid nephrotoxins, optimize hemodynamics

Low Risk (Bottom 70%):
└── Standard care protocols
```

**Resource Allocation**:
- Predict dialysis need → Equipment/staffing preparation
- Predict prolonged ICU → Bed management, capacity planning
- Risk-stratified monitoring intensity → Cost-effective care

---

## ⚠️ Limitations

### Data-Related
1. **Single-center**: MIMIC-IV is from Beth Israel Deaconess Medical Center; generalizability uncertain
2. **Retrospective**: Cannot establish causality; unmeasured confounding possible
3. **Missing data**: 30-50% missingness for some labs (lactate, ABG, bilirubin)
4. **Measurement bias**: Lab ordering reflects clinical suspicion → sicker patients have more data

### Methodological
5. **Temporal simplification**: 24-hour windows aggregate time-varying physiology
6. **Feature selection**: No formal feature selection beyond variance/correlation filters
7. **Class imbalance**: Mortality relatively rare (~12-15%) → lower positive predictive value
8. **Cluster validation**: No external cohort validation of phenotypes

### Clinical
9. **Outcome timing**: Hospital mortality may miss post-discharge deaths
10. **AKI definition**: KDIGO creatinine-based only (no urine output criteria)
11. **Baseline creatinine**: Uses first ICU measurement, may miss pre-admission AKI

---

## 🔮 Future Directions

### Methodological Enhancements
1. **Temporal modeling**: LSTM/Transformer networks to capture physiologic trajectories over time
2. **Causal inference**: Propensity score matching, instrumental variables for treatment effects
3. **Survival analysis**: Time-to-event modeling with competing risks (discharge vs. death)
4. **Uncertainty quantification**: Conformal prediction for reliable confidence intervals
5. **Feature selection**: Recursive feature elimination, LASSO for dimensionality reduction

### Clinical Translation
6. **External validation**: 
   - Test on eICU Collaborative Research Database
   - Validate on MIMIC-III for temporal consistency
   - Partner with institutions for site-specific validation
7. **Prospective validation**: Silent mode deployment, alert fatigue assessment
8. **Decision support integration**: HL7 FHIR-compatible API for EHR embedding
9. **Fairness audit**:
