"""
Configuration file for the Earnings Call Intelligence Engine
"""

from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'
FINAL_DATA_DIR = DATA_DIR / 'final'

SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Data collection parameters
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'
TARGET_COMPANIES = 500  # S&P 500

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# FinBERT parameters
FINBERT_MODEL = "ProsusAI/finbert"
MAX_LENGTH = 512
EMBEDDING_DIM = 768

# Financial features
FINANCIAL_FEATURES_COUNT = 28

# Target parameters
ABNORMAL_RETURN_WINDOW = 3  # days after earnings announcement

# Performance thresholds (from proposal)
PARSING_SUCCESS_THRESHOLD = 0.85
BASELINE_ROC_AUC_THRESHOLD = 0.55
IMPROVEMENT_THRESHOLD = 0.03  # 3% improvement

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Team information
TEAM_MEMBERS = [
    'Shalin N. Bhavsar',
    'Ramanuja M. A. Krishna',
    'Freya H. Shah',
    'Kruthika Suresh',
    'Harihara Y. Veldanda'
]

PROJECT_TITLE = "Earnings Call and Risk Intelligence Engine for Financial Decision Support"
COURSE = "FSE 570"
UNIVERSITY = "Arizona State University"

print(f"✅ Configuration loaded for {PROJECT_TITLE}")