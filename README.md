# Earnings Call and Risk Intelligence Engine

**FSE 570 Capstone Project**  
Arizona State University

## 📋 Project Overview

An AI-powered system that analyzes earnings call transcripts and financial data to predict post-earnings stock performance. The system combines traditional NLP (FinBERT), financial metrics, and agentic AI to generate actionable investment insights.

## 👥 Team Members

- Shalin N. Bhavsar - sbhavsa8@asu.edu
- Ramanuja M. A. Krishna - rkrish79@asu.edu
- Freya H. Shah - fshah14@asu.edu
- Kruthika Suresh - ksures21@asu.edu
- Harihara Y. Veldanda - hveldan1@asu.edu

## 🎯 Problem Statement

Every year, 10,000+ earnings calls generate massive unstructured text data. Manual analysis is time-consuming and doesn't scale. This project develops an automated system to:
- Extract sentiment and risk signals from earnings transcripts
- Integrate unstructured text with structured financial data
- Predict abnormal stock returns post-earnings
- Provide interpretable insights for investment decisions

## 🏗️ Architecture
```
Data Collection → Preprocessing → Feature Engineering → ML Modeling → Prediction
↓                ↓                  ↓                  ↓              ↓
Market Data    - Cleaning        - FinBERT         - XGBoost      ROC-AUC
Financials     - Alignment       - LM Dict         - Random Forest Improvement
Transcripts    - Validation      - Agent AI        - Ensemble     ≥3%
```

## 🚀 Key Features

- **Data Collection**: 503 S&P 500 companies, 1.24M price records, 3.1K quarterly financials, 1,720 transcripts
- **NLP Pipeline**: FinBERT embeddings (768-dim) + sentiment analysis
- **Agentic AI**: Multi-agent system for advanced feature extraction (Phase 9-10)
- **ML Models**: Logistic Regression baseline, XGBoost primary, Random Forest validation
- **Explainability**: SHAP values for feature importance

## 📊 Dataset

| Data Source | Records | Time Period | Status |
|-------------|---------|-------------|--------|
| Stock Prices | 1,244,349 | 2016-2026 | ✅ Complete |
| Financial Statements | 3,125 | 2024-2026 | ✅ Complete |
| Earnings Transcripts | 1,720 | 2016-2026 | ✅ Complete |
| S&P 500 Index | 2,546 | 2016-2026 | ✅ Complete |

## 🛠️ Technology Stack

**Languages & Frameworks:**
- Python 3.12
- PyTorch 2.7
- Transformers (Hugging Face)
- scikit-learn 1.8

**Key Libraries:**
- **NLP**: FinBERT, NLTK, spaCy
- **ML**: XGBoost, LightGBM, scikit-learn
- **Data**: pandas, numpy, yfinance
- **Visualization**: matplotlib, seaborn, plotly
- **Agents**: Anthropic Claude API (Phase 9-11), LangGraph

## 📁 Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── market/           # Stock prices, S&P 500 index, tickers
│   │   ├── financial/        # Income statements, balance sheets, cash flows
│   │   └── transcripts/      # Raw earnings call transcripts (JSON + individual .txt)
│   ├── processed/
│   │   ├── aligned/          # Temporally aligned dataset
│   │   ├── financial/        # Normalized financial features
│   │   ├── market/           # Processed market data
│   │   └── transcripts/      # Cleaned and segmented transcripts
│   ├── features/             # FinBERT and NLP extracted features
│   └── final/                # Model-ready train/test datasets
├── results/
│   ├── agentic_features/     # Agent-discovered features and rankings
│   ├── agentic_models/       # Agent-enhanced model outputs
│   ├── agentic_orchestration/ # LangGraph orchestration logs and reports
│   ├── baseline_models/      # Phase 6 baseline documentation
│   ├── xgboost_models/       # Phase 7 XGBoost results
│   ├── validation/           # Cross-validation, SHAP, error analysis
│   ├── figures/              # Charts and visualizations
│   ├── quality_reports/      # Data quality reports
│   ├── tables/               # Result tables
│   └── reports/              # Final reports
├── src/
│   ├── data_collection/      # Scrapers and collectors
│   ├── preprocessing/        # Data cleaning pipelines
│   ├── features/             # Feature engineering
│   ├── agentic/              # Agentic AI modules (Phase 9-11)
│   ├── models/               # ML model implementations
│   ├── evaluation/           # Metrics and validation
│   └── utils/                # Shared utilities
├── notebooks/                # Jupyter notebooks for EDA
├── models/                   # Saved model files
├── logs/                     # Execution logs
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/earnings-intelligence-engine.git
cd earnings-intelligence-engine
```

2. Running the entire Pipeline

**On Windows:**
```bash
run_everything.bat
```

**On MacOS/Linux:**
```bash
sh run_everything.sh
```

**Or run manually:**

Install dependencies:
```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python verify_installation.py
```

## 🎯 Usage

### Phase 1: Data Collection
```bash
python run_phase1ab.py   # S&P 500 tickers + market data
python run_phase1c.py    # Financial statements
python run_phase1d.py    # Earnings transcripts
```

Verify each phase:
```bash
python verify_phase1ab.py
python verify_phase1c.py
python verify_phase1d.py
```

### Phase 2: Preprocessing
```bash
python run_phase2a.py    # Transcript cleaning
python run_phase2b.py    # Speaker segmentation
python run_phase2c.py    # Financial normalization
python run_phase2d.py    # Temporal alignment
python run_phase2e.py    # Quality reporting
```

### Phase 3: Feature Engineering
```bash
python run_phase3a.py    # FinBERT extraction
python run_phase3b.py    # NLP features (LM dictionary)
python run_phase3c.py    # Feature integration + target enhancement
```

### Phase 4-5: Baseline Modeling & Regression & Classification
```bash
python run_phase4.py
```

### Phase 6: Baseline Models Comparison
```bash
python run_phase6.py
```

### Phase 7: XGBoost Models
```bash
python run_phase7.py
```

### Phase 8: Model Validation & SHAP Analysis
```bash
python run_phase8.py
```

### Phase 9: Agentic AI Feature Engineering
```bash
python run_phase9.py
```

### Phase 10: Agent-Enhanced Modeling
```bash
python run_phase10.py
```

### Phase 11: Agentic AI Orchestration Layer (LangGraph)
```bash
python run_phase11.py
```

## 📈 Results

| Model | Features | Target ROC-AUC | Achieved ROC-AUC | Status |
|-------|----------|----------------|------------------|--------|
| Baseline (Financial Only) | 28 | ≥0.55 | 0.568 | ✅ Complete |
| Enhanced (FinBERT) | 796 | ≥0.58 | — | ✅ Complete |
| Agent-Enhanced | 863 | ≥0.61 | 0.572 | ✅ Complete |

**Final Pipeline Performance (Phase 11 Orchestration):**
- Baseline AUC: **0.568** (exceeds ≥0.55 threshold)
- Final AUC: **0.572**
- Improvement: **+0.004**
- Best Model: `agentic_ai`
- Feature dimensionality: **863** (post-integration)
- Parsing success rate: **87.0%**
- Train/test split: **69.9% / 30.1%** (chronological)

## 🧪 Validation Strategy

- **Train/Test Split**: 70/30 chronological
- **Cross-Validation**: 5-fold time-series CV
- **Metrics**: Accuracy, F1, ROC-AUC
- **Statistical Test**: DeLong test (p < 0.05)
- **Baseline Comparison**: Sentiment-only vs. Financial-only

## 📊 Current Progress

- [x] Phase 0: Environment Setup
- [x] Phase 1A: S&P 500 Tickers
- [x] Phase 1B: Market Data Collection
- [x] Phase 1C: Financial Statements
- [x] Phase 1D: Transcript Collection (1,720 transcripts, 508 tickers)
- [x] Phase 2: Data Preprocessing (2a–2e)
- [x] Phase 3: Feature Engineering (3a–3c)
- [x] Phase 4-8: ML Modeling & Validation
- [x] Phase 9-10: Agentic Feature Engineering & Enhanced Modeling
- [x] Phase 11: LangGraph Orchestration Layer

## 🤖 Agentic AI Enhancement (Phase 9-11)

**Multi-Agent System:**
- **Data Ingestion Agent**: Decides data sources, handles API failures
- **Preprocessing Validation Agent**: Checks parsing quality (≥80% threshold), validates temporal alignment
- **Feature Extraction Agent**: Runs FinBERT (768-dim), falls back to Loughran-McDonald if needed
- **Data Integration Agent**: Validates 863-dimensional feature fusion, enforces train/test temporal split
- **Modeling & Validation Agent**: Establishes baseline (≥0.55 AUC), selects best architecture
- **Reporting Agent**: Generates documentation, logs all autonomous decisions

**Orchestration:** LangGraph-based autonomous pipeline control with quality validation, fallback triggers, and complete reproducibility logging.

**Agent-Derived Features:** Management confidence, competitive pressure, strategic shifts, risk severity

## 📚 References

- Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift
- FinBERT: Financial Sentiment Analysis (Araci, 2019)
- Loughran-McDonald Financial Dictionary (2011)

## 📝 License

This project is part of academic coursework at Arizona State University.

## 🙏 Acknowledgments

- Arizona State University, FSE 570
- Anthropic (Claude API for agentic AI)
- Hugging Face (FinBERT model)

## 📧 Contact

For questions or collaboration:
- sbhavsa8@asu.edu - Shalin N Bhavsar
- rkrish79@asu.edu - Ramanuja M.A Krishna
- fshah14@asu.edu - Freya Shah
- ksures21@asu.edu - Kruthika Suresh
- hveldan1@asu.edu - Harihara Y. Veldanda

---

**Status**: 🟢 Complete
