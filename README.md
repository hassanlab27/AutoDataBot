# AutoDataBot 🧠🤖

**AutoDataBot** is a free, local, open-source, zero-paid-API automated data-science platform designed for small-to-medium tabular CSV datasets.

It provides a transparent, deterministic data-engineering and exploratory analysis pipeline running completely on your local machine—with zero cloud dependencies, zero external LLM API costs, and zero cluster bloat.

---

## Architecture Overview

```
┌──────────────────────────────────────┐       HTTP REST       ┌──────────────────────────────────────┐
│        Frontend (React + Vite)       │ ◄───────────────────► │          Backend (FastAPI)           │
│  - TypeScript & CSS Design Tokens    │       Port 5173       │  - Python 3.12+ (managed by uv)      │
│  - Plotly Interactive Visualizations │       Port 8000       │  - In-Memory Dataset LRU Cache       │
│  - Target-Independent & Aware EDA    │                       │  - Robust Numerical & Cat Analytics  │
│  - Drag & Drop CSV Ingestion         │                       │  - Pearson Correlation & IQR Outliers│
│  - Quality Scorecard & Data Preview  │                       │  - Deterministic Chart Recommender   │
└──────────────────────────────────────┘                       └──────────────────────────────────────┘
```

- **Frontend:** React 18, TypeScript, Vite, Plotly (`plotly.js-dist-min`), Lucide Icons, Glassmorphic Dark UI.
- **Backend:** FastAPI, Uvicorn, Pandas, NumPy, SciPy, Pydantic v2.
- **Storage:** Local filesystem (`data/uploads/`), path-traversal proof, preserves raw CSV untouched.
- **Analytics & Caching:** Pure deterministic Pandas/NumPy/SciPy with an in-memory DataFrame LRU cache.

---

## Phase 2 — EDA & Visualization Capabilities

AutoDataBot delivers automated, readable, and statistically rigorous exploratory analysis before any machine learning modeling.

### 1. Supported Analyses
- **Dataset Overview:** Comprehensive summary metrics (row counts, column counts, exact memory usage, missing cells and percentage, duplicate rows and percentage, constant columns, high-cardinality columns).
- **Numerical Analysis:** Robust descriptive statistics handling `NaN`, `+inf`, `-inf` safely without crashing (Count, Valid Count, Missing Count & %, Mean, Median, Standard Deviation, Min, Max, Q1 [25%], Q3 [75%], IQR, Skewness, Kurtosis).
- **Categorical Analysis:** Cardinality and distribution metrics (Count, Valid Count, Missing Count & %, Unique Count, Mode, Mode %, Top N Categories with exact frequencies, and aggregated 'Other' category).
- **Missing-Value Analysis:** Dataset-wide missingness totals, percentage of all cells missing, columns with missing data, and automatic detection of columns with severe missingness (≥ 50%).
- **Correlation Analysis:** Pearson correlation matrix across numerical features with automatic variance-based ranking if columns exceed limits.
- **Strong Correlations Extraction:** Ranked table of unique feature pairs with absolute Pearson correlation $|r| \ge 0.50$, indicating relationship strength and direction.
- **Outlier Detection:** Transparent Interquartile Range (IQR) detection determining Q1, Q3, IQR, lower fence, upper fence, potential outlier count, and percentage.
- **Datetime & Temporal Analysis:** Min date, max date, date range duration in days, and records over time.
- **Target Selection & Target-Aware EDA:** Defaults to target-independent EDA. When a user selects an optional target column from the dropdown, target distribution, class balance warnings, and top features correlated/associated with the target are computed.

---

### 2. Supported Interactive Chart Types (Plotly)
- **Missing Values Chart:** Interactive horizontal bar chart colored by missingness severity (low, moderate, severe ≥ 50%).
- **Distribution Histogram:** Feature distributions with automatic binning and frequency hover values.
- **Distribution Box Plot:** Quantiles, medians, and whisker fences for outlier inspection.
- **Categorical Bar Chart:** Top 10 categories with frequency bars and percentage hover templates.
- **Correlation Heatmap:** Interactive heatmap with diverging cool-to-warm colorscale ($-1.0$ to $+1.0$) and cell hover values.
- **Scatter Plot with Trendline:** Bivariate numerical scatter plots with linear trendlines.
- **Grouped Box Plot:** Numerical distributions segmented by categorical levels.
- **Contingency Heatmap:** 2D frequency cross-tabulation for categorical-categorical pairs.
- **Time Series Line Chart:** Chronologically sorted temporal records with markers.

---

### 3. Automatic Chart Selection Rules
AutoDataBot uses a deterministic, rule-based recommender (zero LLM latency or hallucination):

| Feature Pair / Selection | Recommended Visualization | Visual Output |
| :--- | :--- | :--- |
| **Numeric + Numeric** | Scatter Plot | Subsampled scatter markers + fitted linear trendline |
| **Categorical + Numeric** | Grouped Box Plot | Value distribution across top categories |
| **Categorical + Categorical**| Contingency Heatmap | Cross-tabulation frequency matrix |
| **Datetime + Numeric** | Time Series Line Chart| Chronological line plot with date axis |
| **Single Numeric** | Histogram + Box Plot | Binned frequency distribution + IQR box plot |
| **Single Categorical** | Horizontal Bar Chart | Top categories with counts and percentages |
| **Single Datetime** | Timeline Histogram | Records count grouped over temporal intervals |

---

### 4. Sampling Behavior
- To guarantee responsive 60fps UI performance and prevent massive payload transfers to the browser, bivariate scatter plots are subsampled to `EDA_MAX_SCATTER_POINTS = 2000`.
- Subsampling is deterministic using a fixed seed (`EDA_RANDOM_SEED = 42`), ensuring reproducibility across identical requests.
- The UI displays an explicit indicator whenever data sampling is active.

---

### 5. Methodological & Epistemological Guardrails

> [!IMPORTANT]
> **Correlation ≠ Causation:**
> Correlation measures statistical linear co-movement only. It does not establish a causal mechanism, nor does it guarantee predictive feature importance. The UI explicitly marks all association metrics with a clear disclaimer.

> [!NOTE]
> **Potential Outlier ≠ Invalid Data:**
> Values outside the IQR fences ($Q_1 - 1.5\text{IQR}$, $Q_3 + 1.5\text{IQR}$) are statistical anomalies, not necessarily corrupted records or measurement errors. AutoDataBot **never modifies, truncates, or deletes data** during the EDA phase.

---

### 6. Configurable EDA Limits (`app/core/config.py`)

All limits are configurable via environment variables (prefixed with `AUTODATABOT_`):

| Setting | Default | Description |
| :--- | :--- | :--- |
| `EDA_MAX_NUMERIC_DISTRIBUTION_COLUMNS` | `10` | Default columns rendered in distribution batches |
| `EDA_MAX_CATEGORICAL_CHART_COLUMNS` | `10` | Max categorical columns visualized simultaneously |
| `EDA_MAX_CATEGORIES_PER_CHART` | `10` | Max individual bars before aggregating into 'Other' |
| `EDA_MAX_CORRELATION_COLUMNS` | `20` | Max columns in correlation heatmap (ranked by variance) |
| `EDA_MAX_SCATTER_POINTS` | `2000` | Max points sampled for bivariate scatter visualizations |
| `EDA_TOP_CORRELATION_PAIRS` | `10` | Max strong correlation pairs shown in summary table |
| `EDA_OUTLIER_IQR_MULTIPLIER` | `1.5` | Standard Tukey IQR fence multiplier |
| `EDA_STRONG_CORRELATION_THRESHOLD` | `0.5` | Minimum absolute correlation threshold for strong pairs |
| `EDA_RANDOM_SEED` | `42` | Seed for deterministic sampling |

---

---

## Phase 3 — Preprocessing & ML Dataset Preparation

AutoDataBot implements a leakage-free, reproducible, and fully serializable preprocessing engine to prepare tabular CSV datasets for downstream AutoML without training models in this phase.

### 1. Preprocessing Pipeline Flow

```text
Raw CSV
   ↓
Validated Dataset
   ↓
User-selected Target Column
   ↓
Conservative Problem-Type Detection (Classification / Regression / Ambiguous)
   ↓
Feature Selection & Leakage Audit (Direct duplicate detection, row ID detection)
   ↓
Strict Feature/Target Segregation: (X ∩ {y} = ∅)
   ↓
Train/Test Partition (80/20 default, Stratified for Classification, Random for Regression)
   ↓
TRAINING DATA ONLY Preprocessing Fitting (Imputation, OneHot, Scaling, Datetime)
   ↓
Hold-out Test Set Transformation (Using already-fitted training statistics)
   ↓
Joblib Serialization (pipeline.joblib, config.json, metadata.json)
   ↓
Processed Training Sample Preview (First 20 records)
```

> [!CRITICAL]
> **Zero Data Leakage Guarantee:**
> **The test set is held out and is not used to fit preprocessing transformations.**
> Imputers (median/mode), scalers (mean/std), and encoders (category vocabularies) are fitted **strictly** on the training partition ($X_{\text{train}}$). The test partition ($X_{\text{test}}$) is only transformed via the fitted pipeline.

---

### 2. Key Phase 3 Capabilities

- **Explicit Target Selection & Validation:** The user explicitly selects the target column. AutoDataBot validates that the target exists, is not empty, contains usable observations, and is not constant.
- **Conservative Problem-Type Detection:** Deterministic rules classify the target into:
  - `binary_classification`: Exactly 2 unique classes or boolean targets.
  - `multiclass_classification`: 3 to 20 unique classes, or low-cardinality discrete strings.
  - `regression`: Continuous numerical float targets or high-cardinality numeric scales.
  - `ambiguous`: Discrete numerical scales (e.g., 1-5 customer rating) where user confirmation is explicitly required.
- **Feature Selection & Quality Audit:**
  - Audits candidate features for constant columns, pairwise exact duplicates, obvious row identifiers (UUIDs, >99% unique keys), free-form text, and high-cardinality categoricals (>50 unique levels).
  - Suspicious columns are flagged with clear human-readable explanations rather than silently dropped.
  - Target Leakage Barrier: Enforces $X_{\text{selected}} \cap \{y\} = \emptyset$ both at UI, service, and router levels.
- **Train/Test Splitting:**
  - Configurable ratio (default: 80% train / 20% test, range: 10% - 40%).
  - Configurable random seed for deterministic reproducibility (default: 42).
  - Stratified split for classification to preserve class balances; randomized split for regression.
  - Verifies disjoint index partitions: $I_{\text{train}} \cap I_{\text{test}} = \emptyset$.
- **Leakage-Resistant Preprocessing:**
  - **Numerical Imputation:** Default median imputation (or mean, most frequent).
  - **Categorical Imputation:** Default most frequent mode (or explicit `__MISSING__` category).
  - **Categorical Encoding:** `OneHotEncoder(handle_unknown="ignore")` to ensure unseen categories in test or future inference data never crash the pipeline.
  - **Infinite Value Sanitizer:** `+inf` and `-inf` are automatically converted to `np.nan` before imputation.
  - **Datetime Component Extractor:** Scikit-learn compatible transformer extracting numeric components (`_year`, `_month`, `_day`, `_dayofweek`) while dropping raw timestamps.
  - **Numerical Scaling:** Configurable `StandardScaler` (zero mean, unit variance) or unscaled pass-through.
- **Artifact Serialization:**
  - Saves the complete fitted pipeline to `outputs/datasets/<dataset_id>/preprocessing/`:
    - `pipeline.joblib`: Serialized ColumnTransformer, encoders, scalers, and datetime transformers.
    - `config.json`: Exact reproducibility configuration (seed, test size, imputation, encoding choices).
    - `metadata.json`: Feature counts, one-hot generated names, and train/test partition distribution summaries.
- **Processed Data Preview:** Live endpoint returning the first 20 records of the transformed training matrix for immediate UI inspection.

---

### 3. Preprocessing REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/datasets/{id}/preprocessing/target-inspect` | Validates target, detects problem type, and returns diagnostics |
| `POST` | `/api/datasets/{id}/preprocessing/features-inspect` | Audits features for flags, duplicates, IDs, and leakage |
| `POST` | `/api/datasets/{id}/preprocessing/prepare` | Executes split, fits pipeline on train, transforms, and saves artifacts |
| `GET`  | `/api/datasets/{id}/preprocessing/summary` | Retrieves metadata of prepared dataset and partition info |
| `GET`  | `/api/datasets/{id}/preprocessing/preview` | Returns a sample preview of the processed training matrix |

---

## Local Setup & Installation

### 1. Setup Backend

```bash
cd backend

# Create virtual environment and install dependencies with uv
uv venv
uv pip install -e ".[dev]"
```

### 2. Setup Frontend

```bash
cd ../frontend
npm install
```

---

## Running Locally

Open two terminal windows:

### Terminal 1 — Backend (FastAPI)

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* Backend API: `http://127.0.0.1:8000`
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### Terminal 2 — Frontend (Vite React)

```bash
cd frontend
npm run dev
```
* Frontend Application: `http://localhost:5173`

---

## Automated Testing Suite

Run the full pytest suite:

```bash
cd backend
.venv/bin/pytest backend/tests
```

**65 automated tests passing:**
- **Phase 1 Baseline (17 tests):** Upload validation, CSV reader, encoding/delimiter detection, schema inspection, data quality scoring, health check.
- **Phase 2 EDA Core (23 tests):** Numerical statistics (handling NaN, inf, all-NaN, constants, single-row), categorical stats, missing value percentages and severity, Pearson correlation matrix, strong pairs extraction, IQR outlier fences, deterministic chart recommendations, all API endpoints.
- **Phase 2 Benchmark Verification (4 tests):** Dataset A (numerical), Dataset B (mixed), Dataset C (stress), Dataset D (temporal).
- **Phase 3 Preprocessing Unit & Integration (16 tests):** Target validation, binary/multiclass/regression/ambiguous detection, feature audit flags, leakage barrier enforcement, train/test split disjointness, **mandatory zero-leakage fitting verification**, datetime component extraction, infinite numbers sanitization, joblib serialization, full REST API route integration.
- **Phase 3 Benchmark Verification (5 tests):**
  - **Dataset A:** Pure numerical binary classification.
  - **Dataset B:** Mixed categorical/numerical classification with one-hot expansion.
  - **Dataset C:** Continuous target regression with skewness and outliers.
  - **Dataset D:** Missing values, infinite numbers, and unseen categories in test set.
  - **Dataset E:** Datetime feature extraction combined with categorical and numerical features.

Frontend build & type-checking verification:

```bash
cd frontend
npm run build
```
*(Transpiles TypeScript and bundles React + Plotly chunks with 0 errors in ~27s).*

---

## Development Roadmap

- [x] **Phase 1: Foundation, Ingestion & Data Quality**
- [x] **Phase 2: Exploratory Data Analysis (EDA) & Visualization** *(Completed)*
- [x] **Phase 3: Preprocessing & ML Dataset Preparation** *(Completed)*
- [ ] **Phase 4: AutoML Model Training**
- [ ] **Phase 5: Evaluation & Explainability**
- [ ] **Phase 6: Reporting & One-Click Export**
