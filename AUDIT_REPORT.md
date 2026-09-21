# Comprehensive Project Audit: AutoDataBot
**Audit Date:** September 2026  
**Audited Directory:** `/home/hassan/Devalopment folder/AutoDataBot`  
**Repository Branch:** `main` (commit `2dd0b3a`)  
**Audit Scope:** 100% Full-Stack Codebase Audit (Architecture, Code Quality, Dependencies, Security, Runtime Culprits, and UI/UX)

---

## 1. Executive Summary & Health Scorecard

**AutoDataBot** is conceived as an end-to-end data analysis, Automated Machine Learning (AutoML), and Retrieval-Augmented Generation (RAG) conversational assistant. It provides a web interface (Flask) where users can upload datasets (CSV) for automated profiling, visualization, and model training (via TPOT and scikit-learn SGD), or ask questions against indexed domain documents (via FAISS + SentenceTransformers) and Wikipedia.

### Overall System Health: **CRITICAL / FAILING (Score: 32/100)**

| Dimension | Rating | Status | Primary Finding |
| :--- | :---: | :---: | :--- |
| **Runtime Viability** | **FAIL** | 🔴 Critical | The application cannot start in a clean environment due to uninstalled heavy dependencies (`tpot`, `torch`, `faiss`) and crashes immediately on start or during analysis. |
| **AutoML Architecture** | **BROKEN** | 🔴 Critical | Architectural bait-and-switch: TPOT runs an expensive genetic pipeline search on a sample, but the discovered pipeline is **completely discarded** and replaced by a barebones linear SGD model. |
| **Data Integrity** | **BROKEN** | 🔴 Critical | Categorical imputation bug converts `NaN` to `"nan"` strings, destroying missing value imputation. Multi-chunk training fits a new `OrdinalEncoder` every chunk, rendering categorical encodings meaningless. |
| **Security & Safety** | **HIGH RISK**| 🟠 Vulnerable | Arbitrary file download via path traversal (`/download/<zipname>`), unvalidated 10 GB file uploads, Flask debug mode exposed on `0.0.0.0`. |
| **State & Database** | **DEGRADED** | 🟡 Warning | Every analysis run is inserted **twice** into SQLite. Timestamps are written as `NULL` due to key mismatch (`time` vs `timestamp`). Chat history is never stored. |
| **Frontend & UI/UX** | **DEFECTIVE** | 🟡 Warning | Missing CSS classes referenced in JS (`.checkmark`, `.meta`), `.xlsx` accepted by file picker but crashes backend, no `Enter` key support, no error recovery. |
| **Git Hygiene** | **POOR** | 🔴 Critical | No `.gitignore`. Over 23 MB of binaries, uploaded CSVs, `.joblib` models, SQLite DBs, and `.zip` archives are tracked directly in Git history. |

---

## 2. Technology Stack & Dependency Audit

The project defines dependencies via `requirements.txt` containing **119 pinned packages** from a `pip freeze`.

### Core Technology Stack Components

| Layer | Technology | Declared Version | Purpose in Codebase |
| :--- | :--- | :--- | :--- |
| **Web Framework** | Flask | `3.1.2` | REST API routes (`/`, `/chat`, `/analyze`, `/download`) |
| **WSGI / Utilities** | Werkzeug | `3.1.5` | Request handling, secure filename utils |
| **Templating** | Jinja2 | `3.1.6` | Server-rendered HTML (`templates/index.html`) |
| **AutoML Search** | TPOT | `1.1.0` | Genetic algorithm ML pipeline optimization |
| **Machine Learning** | scikit-learn | `1.8.0` | Preprocessing, SGDClassifier, SGDRegressor, metrics |
| **Data Profiling** | ydata-profiling | `4.18.1` | Automated HTML exploratory data profiling reports |
| **Vector Indexing** | faiss-cpu | `1.13.2` | Vector similarity search for question answering |
| **Embeddings** | sentence-transformers | `5.2.0` | `all-MiniLM-L6-v2` dense text embedding generation |
| **Knowledge Base** | wikipedia | `1.4.0` | Dynamic Wikipedia article scraping |
| **Data Manipulation**| pandas, numpy | `2.3.3`, `2.3.5` | Tabular data processing and chunked streaming |
| **Visualization** | matplotlib, seaborn | `3.10.0`, `0.13.2` | Heatmaps and distribution histograms |
| **Model Persistence**| joblib | `1.5.3` | Serializing trained estimator artifacts |
| **Database** | SQLite3 | Python Built-in | Persisting run history in `memory_store.db` |

### Dependency Issues & Redundancies

1. **Extreme Dependency Bloat (CUDA in CPU setup):**
   Although `faiss-cpu` is specified, `requirements.txt` includes 15+ heavy NVIDIA CUDA 12 packages:
   - `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cusolver-cu12`, `triton`
   This inflates environment installation size by **over 3.5 GB** without providing any GPU acceleration to FAISS.
2. **Missing Core Packages in Active Environment:**
   Running `python3 app.py` on the active machine fails immediately:
   ```
   ModuleNotFoundError: No module named 'tpot'
   ```
3. **Dead / Unused Dependencies:**
   `dask`, `dask-expr`, `dask-jobqueue`, `distributed`, `alembic`, `SQLAlchemy`, `wordcloud`, `ImageHash`, `puremagic`, `typeguard`, and `optuna` are installed and declared in `requirements.txt` but **never imported** anywhere in `app.py`, `core/`, or `scripts/`.

---

## 3. Directory & File Structure Audit

```
AutoDataBot/
├── app.py                     # [CORE ENTRY] Flask application server and routing
├── requirements.txt           # [DEPS] 119 pinned dependencies (frozen environment dump)
├── README.md                  # [DOCS] Outdated setup guide and inconsistent directory diagram
├── memory_store.db            # [DB] Active SQLite database committed to Git (Anti-pattern)
├── core/                      # [BACKEND ENGINE MODULES]
│   ├── __init__.py            # Package initialization (empty)
│   ├── config.py              # Configuration constants (DEAD CODE: never imported)
│   ├── utils.py               # Problem type detection heuristic (flawed logic)
│   ├── data_cleaner.py        # Data cleaning & encoding (CRITICAL imputation bug)
│   ├── visualizer.py          # Matplotlib/Seaborn charting (Missing Agg backend & path bugs)
│   ├── automl_engine.py       # TPOT discovery & incremental SGD training (Bait-and-switch)
│   ├── memory_manager.py      # SQLite repository (timestamp NULL bug, chat unused)
│   └── chatbot_engine.py      # FAISS + Wikipedia engine (Missing load_index method)
├── data/                      # [KNOWLEDGE ASSETS]
│   ├── faiss_index/           # Dual conflicting index formats (npy vs jsonl, L2 vs IP)
│   └── wiki_pages/            # Cached Wikipedia text dumps
├── scripts/                   # [OFFLINE INGESTION SCRIPTS]
│   ├── build_wiki_dataset.py  # Ingestion script A (saves wiki_index.faiss + .npy files)
│   └── ingest_wiki.py         # Ingestion script B (saves faiss.index + .jsonl files)
├── templates/
│   └── index.html             # Single-page interface template
├── static/
│   ├── script.js              # Frontend client script (Missing error recovery & listeners)
│   └── style.css              # Minimal dark styling (Missing classes referenced in JS)
├── uploads/                   # [COMMITTED UPLOADS] 18 raw CSV files tracked in Git history
└── projects/                  # [COMMITTED RUNS] Generated models, zips, HTML reports in Git
```

---

## 4. Deep Line-by-Line Code Audit & Culprit Inventory

### File 1: `app.py`

| Line(s) | Code Snippet / Construct | Severity | Culprit & Vulnerability Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **17** | `app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * ...` | 🟠 High | **10 GB Upload Cap without Streaming**: Allows massive unbuffered uploads that can saturate server memory and exhaust disk storage (Denial of Service). | Lower limit to reasonable value (e.g. 100 MB) or implement chunked stream writing. |
| **25-28** | `general_index_dir=os.path.join(BASE_DIR, "data", "faiss_general")` | 🟡 Medium | `data/faiss_general` does not exist in the repository; engine attempts to create and reference non-existent directory. | Standardize index directories to a unified, consistent location. |
| **30-36** | `chat_engine.load_index(INDEX_DIR)` | 🔴 **CRITICAL** | **Method Does Not Exist**: `ChatbotEngine` has no `load_index()` method (only internal `_load_index(target)`). When the app boots, this always throws `AttributeError: 'ChatbotEngine' object has no attribute 'load_index'`. | Implement a public `load_index()` or let `__init__` handle loading properly. |
| **48-56** | `reply = chat_engine.answer(message, top_k=5)` | 🔴 **CRITICAL** | **Dead Wikipedia Fallback**: `chat_engine.answer` defaults to `target="dataset"`. The dynamic Wikipedia retrieval logic only executes when `target == "general"`. As a result, the chatbot **never** uses Wikipedia. | Support passing `target` or auto-detect whether question pertains to dataset vs general knowledge. |
| **59-88** | `def analyze(): ... auto_engine.run_pipeline(...)` | 🔴 **CRITICAL** | **Synchronous Blocking in Web Request**: Running TPOT genetic optimization (5 generations, 30 population on 100,000 rows) inside a synchronous Flask request blocks the worker for up to hours, triggering HTTP 504 Gateway Timeouts. | Run analysis as an asynchronous background job (e.g. Celery, RQ, or `concurrent.futures`) with polling/SSE. |
| **61 vs 98** | `from werkzeug.utils import secure_filename` vs `def secure_filename(name: str)` | 🟡 Medium | **Shadowed / Duplicate Definition**: Line 61 imports `secure_filename` from Werkzeug inside the function, while lines 98-99 define a custom top-level `secure_filename`. Redundant and confusing. | Standardize on `werkzeug.utils.secure_filename` and delete lines 98-99. |
| **63-69** | `file.save(path)` | 🟠 High | **No MIME / Extension Validation**: The server accepts any file type without validating file magic or extension. Uploading an executable, script, or non-tabular file proceeds to model execution and crashes. | Validate file extension (`.csv`) and content headers before saving. |
| **74** | `datetime.datetime.utcnow().strftime(...)` | 🟢 Low | **Python 3.12+ Deprecation Warning**: `utcnow()` is deprecated; produces runtime warnings on modern Python. | Use `datetime.datetime.now(datetime.timezone.utc)`. |
| **79 vs 100**| `memory.add_run(summary)` in `app.py` and `self.memory.add_run(summary)` in `automl_engine.py` | 🟡 Medium | **Double Insertion Bug**: Every single run is inserted twice into the SQLite `runs` table. | Remove the redundant call in `app.py`. |
| **90-95** | `p = os.path.join(PROJECTS, zipname)` | 🟠 **HIGH RISK** | **Path Traversal Vulnerability (CWE-22)**: `os.path.join` with user-supplied URL parameters can traverse paths if not sanitized via `werkzeug.utils.safe_join`. Furthermore, any file inside `projects/` (not just `.zip`) can be exfiltrated. | Use `werkzeug.utils.safe_join(PROJECTS, zipname)` and assert `zipname.endswith('.zip')`. |
| **102-103**| `app.run(debug=True, host="0.0.0.0", port=5000)` | 🟠 High | **Insecure Deployment Configuration**: Running with `debug=True` bound to `0.0.0.0` exposes Werkzeug's interactive PIN debugger to the entire network, enabling arbitrary remote code execution (RCE). | Disable `debug=True` in production, bind to `127.0.0.1` or serve via Gunicorn/Uvicorn. |

---

### File 2: `core/config.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **1-5** | `CHUNK_ROWS = 200_000`<br>`PROFILE_SAMPLE = 50_000`<br>`SMALL_DATA_THRESHOLD = 500_000`<br>`TFIDF_SAMPLE = 100_000`<br>`RANDOM_STATE = 42` | 🟡 Medium | **100% Dead Code**: Not a single file in the entire repository imports `core.config`. `AutoMLEngine` hardcodes its own values (`CHUNK_SIZE = 200_000`, `n=100_000`, `n=50_000`, `random_state=42`). Changes to `config.py` have zero effect. | Import and use configuration parameters consistently across `core/`. |

---

### File 3: `core/utils.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **4-13** | `def detect_problem_type(y):` | 🔴 **CRITICAL** | **Flawed Problem Heuristic**: Checks `len(np.unique(y)) < 20`.<br>1. Regression datasets with few rows or discrete values (e.g. counts < 20) are misclassified as classification.<br>2. Classification datasets with > 20 classes (e.g. 50 US States, categories) are misclassified as regression.<br>3. Does not account for `category`, `boolean`, or pandas `StringDtype`. | Check dtype, target cardinality relative to sample size, or allow user override via the UI. |

---

### File 4: `core/data_cleaner.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **24-27** | `if df[col].dtype == object: df[col] = df[col].astype(str)` | 🔴 **CRITICAL** | **Catastrophic Imputation Failure**: Converting object columns with `NaN` to `str` transforms `np.nan` into string `"nan"`. Then, lines 35-36 (`df[c].fillna(...)`) **never execute** because there are no longer nulls. Missing values become permanent `"nan"` category strings. | Clean and impute missing values **before** string coercion, or replace `"nan"` / `"None"` strings back to `np.nan`. |
| **32-34** | `df[c] = df[c].fillna(df[c].median())` | 🟡 Medium | If a numeric column is 100% null, `median()` returns `np.nan`, leaving unhandled nulls that crash downstream estimators. | Add fallback imputation value (e.g. 0.0) if median is NaN. |
| **41-53** | `label_encode(self, X_train, X_test, cat_cols)` | 🟡 Medium | Modifies `X_train` and `X_test` slices directly, triggering pandas `SettingWithCopyWarning`. Also, this helper method is defined on `DataCleaner` but **never called** by `AutoMLEngine`. | Clean up unused methods or integrate properly. |

---

### File 5: `core/visualizer.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **2-3** | `import matplotlib.pyplot as plt` | 🟠 High | **Missing Headless Backend**: `matplotlib.use('Agg')` is not set. In headless server environments (e.g. Linux Docker/VMs without X11), this can cause crashes when creating plots. | Insert `import matplotlib; matplotlib.use('Agg')` before `pyplot` import. |
| **14** | `sns.heatmap(df.select_dtypes(include=["number"]).corr(), ...)` | 🟡 Medium | If a dataset has fewer than 2 numeric columns or all constant columns, `corr()` produces empty or all-NaN matrices, leading to broken heatmaps or runtime warnings. | Guard with `if len(num_cols) >= 2:`. |
| **28** | `p = os.path.join(self.plots_dir, f"{c}_hist.png")` | 🔴 **CRITICAL** | **Path Error on Slashed Column Names**: If a column name has `/` (e.g. `unit/price`, `rev/cost`), `os.path.join` interprets it as a subfolder, crashing with `FileNotFoundError`. | Sanitize column names before formatting filenames (e.g. `re.sub(r'[^a-zA-Z0-9_-]', '_', c)`). |

---

### File 6: `core/automl_engine.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **67-73** | Phase 1 TPOT vs Phase 2 SGD | 🔴 **CRITICAL** | **Architectural Bait-and-Switch**: Phase 1 spends minutes/hours evolving a TPOT pipeline (RandomForest, XGBoost, etc.) and extracts `best_strategy = str(automl.fitted_pipeline_)`. Then Phase 2 **completely ignores** the pipeline and fits a rudimentary linear `SGDClassifier`/`SGDRegressor`. The user receives SGD, not the discovered pipeline! | Either serialize and return the fitted TPOT model, or use the discovered pipeline architecture in downstream training. |
| **182-185**| `enc = OrdinalEncoder(...)`<br>`X[cat_cols] = enc.fit_transform(X[cat_cols])` | 🔴 **CRITICAL** | **Inconsistent Categorical Encoding per Chunk**: A new `OrdinalEncoder` is instantiated and fit on **every single chunk**. Category "A" might map to 0 in chunk 1, but map to 3 in chunk 2. The SGD model is fed corrupted, unstable feature encodings across chunks. | Pre-scan categories globally or use hashing vectorizers/one-hot encoders with a fixed vocabulary. |
| **187-190**| `if first_chunk and num_cols: scaler.fit(X[num_cols])` | 🟡 Medium | `first_chunk = False` is set only on line 202 inside the `else` (classification) branch. In **regression mode**, `first_chunk` is **never** set to `False`, causing `scaler.fit` to re-fit and overwrite itself on every chunk. | Set `first_chunk = False` outside the if/else block. |
| **209-217**| Validation Phase in `_train_full_incremental` | 🔴 **CRITICAL** | **Validation Crash on Categorical Features**: Validation data `Xv` is extracted from `val`, but `OrdinalEncoder` and `scaler` are **never applied** to `Xv`! Calling `model.predict(Xv)` immediately crashes with `ValueError: could not convert string to float` on any dataset with text columns. | Apply fitted encoders and scaler to validation features prior to calling `model.predict()`. |
| **165** | `full_target = pd.read_csv(path, usecols=[target])` | 🟡 Medium | **Memory Inefficiency**: Reads the entire target column into memory at once, which undermines the chunked memory advantage on massive multi-gigabyte datasets. | Stream target values or determine unique classes during chunk passes. |

---

### File 7: `core/memory_manager.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **39-40** | `summary.get("time")` | 🔴 **CRITICAL** | **Missing Timestamps (Database Corruption)**: `add_run()` accesses `summary.get("time")`. However, `automl_engine.py` writes `"timestamp"` into the summary dict. Result: all recent runs in `memory_store.db` have `timestamp = NULL`. | Change to `summary.get("timestamp") or summary.get("time")`. |
| **53-58** | `add_chat(self, user_text, bot_text)` | 🟡 Medium | **Unused Database Table**: `add_chat()` is defined to persist conversations in `chat_history`, but `app.py` **never calls** it. The table remains 100% empty. | Call `memory.add_chat(message, reply)` inside the `/chat` route in `app.py`. |

---

### File 8: `core/chatbot_engine.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **37, 45** | `self._load_index("dataset")`<br>`self._load_index("general")` | 🟡 Medium | If indices are missing, internal state defaults silently without notifying caller. | Log clear diagnostic status. |
| **N/A** | Missing `load_index()` | 🔴 **CRITICAL** | `app.py` line 33 calls `chat_engine.load_index(INDEX_DIR)`. This public method does not exist in `ChatbotEngine`. | Add public method: `def load_index(self, directory: str): ...`. |
| **133-145**| `fetch_wikipedia_content` | 🟡 Medium | **Silent Exception Swallow**: `except: pass` catches and discards `wikipedia.DisambiguationError` and network timeouts. When a topic is ambiguous, it returns empty docs instead of picking the first option. | Catch `wikipedia.DisambiguationError`, pick options, and log actual errors. |
| **89-91** | `faiss.IndexFlatIP(self.dimension)` | 🟡 Medium | Uses Inner Product index with normalized embeddings, while `build_wiki_dataset.py` uses `IndexFlatL2`. Discrepancy between indexing scripts. | Standardize indexing metric across all scripts. |

---

### File 9: `scripts/build_wiki_dataset.py` & `scripts/ingest_wiki.py`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **`build_wiki_dataset.py`: 99-103** | Saves `wiki_index.faiss`, `wiki_docs.npy`, `wiki_meta.npy` | 🔴 **CRITICAL** | **Conflicting Output Formats**: `build_wiki_dataset.py` saves `.npy` metadata and `wiki_index.faiss`. However, `chatbot_engine.py` expects `faiss.index` and `metadata.jsonl`. Running `build_wiki_dataset.py` leaves `ChatbotEngine` unable to find or load the index! | Refactor both scripts to use `ChatbotEngine.build_index()` directly. |
| **`build_wiki_dataset.py`: 14-15** | Hardcoded relative paths `"data/wiki_pages"`, `"data/faiss_index"` | 🟡 Medium | Running the script from any directory other than the project root fails with missing directory errors. | Resolve paths relative to `__file__`. |

---

### File 10: `templates/index.html`, `static/script.js` & `static/style.css`

| Line(s) | Construct | Severity | Culprit & Analysis | Recommended Fix |
| :--- | :--- | :---: | :--- | :--- |
| **`script.js`: 60** | `input.accept = '.csv, .xlsx';` | 🔴 **CRITICAL** | **Frontend/Backend Mismatch**: The UI permits users to pick `.xlsx` Excel spreadsheets. When sent to `/analyze`, `pd.read_csv` fails with a decoding error because Excel files are binary zip-compressed XMLs. | Either support `pd.read_excel()` in backend or restrict `input.accept = '.csv'`. |
| **`script.js`: 86-90** | `meta.classList.add('success');`<br>`meta.innerHTML = '<span class="checkmark"></span> Uploaded';` | 🟡 Medium | **Missing CSS Selectors**: `style.css` defines zero rules for `.meta`, `.meta.success`, or `.checkmark`. Checkmark fails to display visually. | Add corresponding styling in `style.css`. |
| **`script.js`: 36-55** | `sendBtn.onclick = async () => { ... }` | 🟡 Medium | **No Keyboard Support**: There is no `keypress` or `keydown` event listener for `Enter`. Users are forced to manually click the send button `➤`. | Add `userInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendBtn.click(); });`. |
| **`script.js`: 45-53, 78-84**| `fetch('/chat')` and `fetch('/analyze')` | 🟡 Medium | **Uncaught Network Errors**: Neither fetch call has `try ... catch`. If the backend throws a 500 error or times out, `hideTyping()` is never reached and the UI is permanently stuck with bouncing typing dots. | Wrap fetches in `try/catch/finally` blocks and call `hideTyping()` in `finally`. |
| **`script.js`: 54** | `appendMessage('bot', escapeHtml(data.response));` | 🟢 Low | **Raw Markdown Render**: Chatbot responses are escaped into plain text, so markdown lists, bold text, and code snippets render as ugly raw markdown. | Include a lightweight markdown renderer (e.g. `marked.js`). |

---

## 5. Security & Vulnerability Analysis

```
+---------------------------------------------------------------------------------------+
|                                SECURITY AUDIT FINDINGS                                 |
+-------------------+--------------------+-----------------+----------------------------+
| Vulnerability     | CWE ID             | Risk Level      | Endpoint / Component       |
+-------------------+--------------------+-----------------+----------------------------+
| Path Traversal    | CWE-22             | HIGH (CVSS 7.5) | GET /download/<zipname>    |
| Denial of Service | CWE-400 / CWE-770  | HIGH (CVSS 7.2) | POST /analyze (10GB file)  |
| Remote Debugger   | CWE-489            | HIGH (CVSS 8.1) | app.run(debug=True, 0.0.0) |
| Missing Auth/Rate | CWE-306 / CWE-799  | MEDIUM          | All API endpoints          |
| Data Exposure     | CWE-200            | MEDIUM          | SQLite DB in Git repo      |
+-------------------+--------------------+-----------------+----------------------------+
```

1. **Path Traversal in `/download/<zipname>` (CWE-22):**
   ```python
   p = os.path.join(PROJECTS, zipname)
   if os.path.exists(p):
       return send_file(p, as_attachment=True)
   ```
   If `zipname` contains path traversal sequences (such as `..%2F`), an attacker could attempt to read arbitrary system files or database files.
2. **Denial of Service via Heavy Computation (CWE-400):**
   An unauthenticated user can trigger an unlimited number of concurrent `run_pipeline` executions, consuming 100% of server CPU/RAM with TPOT genetic runs and crashing the host.
3. **Debug Server on Public Interface (CWE-489):**
   `app.run(debug=True, host="0.0.0.0", port=5000)` allows remote network clients to view execution tracebacks and potentially access the interactive Werkzeug console pin.

---

## 6. Git Hygiene & Repository Bloat

The repository has **no `.gitignore`** file. As a result, transient outputs, binary models, and test datasets have accumulated directly in Git tracking:

- **18 CSV files** in `uploads/` (totaling ~15 MB).
- **11 Project Runs** in `projects/` containing `.joblib` model dumps, `.zip` archives, and HTML reports.
- **2 FAISS index bundles** in `data/faiss_index/` (including 500 KB `.npy` matrices and `.faiss` indices).
- **Production SQLite database** `memory_store.db` tracked in Git.

### Recommended `.gitignore`:
```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Environments
.env
.venv
.bot/
env/
venv/

# Application storage & artifacts
uploads/*
!uploads/.gitkeep
projects/*
!projects/.gitkeep
memory_store.db

# Vector indices and scraped caches
data/faiss_index/*
!data/faiss_index/.gitkeep
data/faiss_general/*
!data/faiss_general/.gitkeep
data/wiki_pages/*
!data/wiki_pages/.gitkeep

# OS / Editor files
.DS_Store
.vscode/
.idea/
```

---

## 7. Complete 0-to-100 Remediation Roadmap

### Phase 1: Critical Stabilization (Runtime & Crash Fixes)
1. **Fix `ChatbotEngine` Missing Method:**
   Implement `load_index(self, directory)` in `core/chatbot_engine.py` so `app.py` boots without throwing `AttributeError`.
2. **Fix Validation Crash in `AutoMLEngine`:**
   Ensure `Xv` undergoes identical categorical encoding and feature scaling before calling `model.predict(Xv)`.
3. **Fix Data Cleaner Imputation:**
   Perform median/mode imputation **before** converting columns to string representations so nulls are not converted to `"nan"`.
4. **Fix Timestamp Null Bug in `MemoryManager`:**
   Map `summary.get("timestamp") or summary.get("time")` in `add_run()`. Remove duplicate `add_run` call in `app.py`.

### Phase 2: Architectural Alignment & Logic Repair
5. **Resolve AutoML Bait-and-Switch:**
   Provide clear pipeline generation: if TPOT is used, save the fitted TPOT pipeline via joblib; if incremental SGD is needed for large datasets, apply a consistent incremental encoder (e.g. `HashingVectorizer` or predetermined vocabulary) rather than re-fitting `OrdinalEncoder` every chunk.
6. **Activate Chatbot Wikipedia Fallback:**
   Allow the user or classifier to route queries between dataset vs. general knowledge (`target="dataset"` vs `target="general"`).
7. **Unify FAISS Ingestion Scripts:**
   Deprecate `build_wiki_dataset.py` in favor of `scripts/ingest_wiki.py` to maintain consistent index (`faiss.index`) and metadata (`metadata.jsonl`) formats.

### Phase 3: Security & Performance Hardening
8. **Secure Download Route:**
   Use `werkzeug.utils.safe_join(PROJECTS, zipname)` and require `.zip` extension.
9. **Async Background Processing:**
   Move `run_pipeline` to a background worker thread (`ThreadPoolExecutor`) with a status/polling endpoint (`/status/<task_id>`).
10. **Sanitize Plotting Column Names:**
    Cleanse special characters from column names before saving histogram plots. Set `matplotlib.use('Agg')`.

### Phase 4: Frontend & UI/UX Polish
11. **Add Missing CSS Rules:**
    Style `.meta`, `.meta.success`, and `.checkmark` in `static/style.css`.
12. **Add Enter Key Support:**
    Enable pressing `Enter` to submit chat prompts.
13. **Robust Error Handling:**
    Add `try...catch...finally` blocks around fetch requests in `static/script.js` to clear typing indicators on failure.
14. **Restrict File Picker:**
    Change `accept=".csv"` to match backend CSV parser capabilities.

### Phase 5: Repository Cleanup
15. **Untrack Artifacts & Add `.gitignore`:**
    Purge committed CSVs, `.zip`, and `.joblib` files from Git tracking.
16. **Prune `requirements.txt`:**
    Separate CPU vs GPU dependencies; eliminate unused packages (`dask`, `optuna`, `alembic`, etc.).
