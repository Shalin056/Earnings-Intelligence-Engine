"""
Phase 7: XGBoost Models
Advanced gradient boosting for improved performance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR


class XGBoostModels:
    """
    XGBoost models for classification and regression
    """
    
    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR / 'xgboost_models'
        self.log_file = LOGS_DIR / 'models' / 'xgboost_models.log'
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
    
    def load_data(self):
        """Load train and test datasets"""
        print("="*60)
        print("PHASE 7: XGBOOST MODELS")
        print("="*60)
        print()
        print("📂 LOADING DATA...")
        
        train_df = pd.read_csv(self.data_dir / 'train_data.csv')
        test_df = pd.read_csv(self.data_dir / 'test_data.csv')
        
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        print(f"✅ Train: {len(train_df)} samples")
        print(f"✅ Test: {len(test_df)} samples")
        print(f"✅ Features: {len(feature_info['feature_columns'])}")
        print()
        
        return train_df, test_df, feature_info
    
    def get_feature_sets(self, feature_info, train_df):
        """Define feature sets"""
        all_cols = set(feature_info['feature_columns']) & set(train_df.columns)
        
        # Financial features
        financial = sorted([c for c in all_cols
                          if c.startswith('financial_') or c.startswith('fin_')])
        
        # Sentiment/NLP features
        sentiment = sorted([c for c in all_cols
                          if any(c.startswith(p) for p in [
                              'sentiment_', 'mgmt_sentiment_', 'analyst_sentiment_',
                              'lm_', 'mgmt_lm_', 'analyst_lm_',
                              'word_count', 'question_count', 'lexical_diversity',
                              'mgmt_word_count', 'analyst_word_count',
                              'lm_sentiment_divergence', 'uncertainty_divergence'])])
        
        # Combined (no embeddings for baseline comparison)
        combined = sorted([c for c in all_cols if not c.startswith('embedding_')])
        
        # Full (with FinBERT embeddings)
        full = sorted(list(all_cols))
        
        return {
            'financial_only': financial,
            'sentiment_only': sentiment,
            'combined': combined,
            'full_with_finbert': full
        }
    
    def clean_and_scale(self, X_train, X_test):
        """Robust data cleaning and scaling"""
        # Handle inf/nan
        for X in [X_train, X_test]:
            X[np.isinf(X)] = np.nan
        
        medians = np.nanmedian(X_train, axis=0)
        for X in [X_train, X_test]:
            inds = np.where(np.isnan(X))
            X[inds] = np.take(medians, inds[1])
        
        # Clip extremes
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test = np.clip(X_test, -1e6, 1e6)
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
    def prepare(self, train_df, test_df, cols):
        """Prepare feature matrices"""
        X_train = train_df[cols].values.astype(np.float64)
        X_test = test_df[cols].values.astype(np.float64)
        return self.clean_and_scale(X_train, X_test)
    
    def train_xgboost_classifier(self, X_train, y_train, X_test, y_test, use_gpu=False):
        """Train XGBoost classifier - tuned for small dataset (901 samples)"""
        n_features = X_train.shape[1]
        # colsample_bytree scales with feature count to avoid overfitting
        colsample = min(0.8, max(0.3, 50 / n_features))

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 3,           # shallow trees prevent overfitting on small data
            'learning_rate': 0.05,    # slower learning, more robust
            'n_estimators': 300,
            'subsample': 0.7,
            'colsample_bytree': colsample,
            'min_child_weight': 10,   # high value forces broader splits
            'gamma': 1.0,             # stronger pruning
            'reg_alpha': 1.0,         # L1 regularization
            'reg_lambda': 5.0,        # L2 regularization
            'scale_pos_weight': sum(y_train == 0) / max(sum(y_train == 1), 1),
            'random_state': 42,
            'n_jobs': -1
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return model, results
    
    def train_xgboost_regressor(self, X_train, y_train, X_test, y_test, use_gpu=False):
        """Train XGBoost regressor - tuned for small dataset"""
        n_features = X_train.shape[1]
        colsample = min(0.8, max(0.3, 50 / n_features))

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.7,
            'colsample_bytree': colsample,
            'min_child_weight': 10,
            'gamma': 1.0,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'random_state': 42,
            'n_jobs': -1
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        results = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'direction': float(np.mean((y_test > 0) == (y_pred > 0)))
        }
        
        return model, results
    
    def run_experiment(self, label, feature_key, train_df, test_df, feature_sets):
        """Run XGBoost experiment for one feature set"""
        print("="*60)
        print(label)
        print("="*60)
        
        cols = feature_sets[feature_key]
        print(f"   Features: {len(cols)}")
        print()
        
        # Prepare data
        X_train, X_test = self.prepare(train_df, test_df, cols)
        
        y_train_clf = train_df['label_binary'].values
        y_test_clf = test_df['label_binary'].values
        y_train_reg = train_df['abnormal_return'].values
        y_test_reg = test_df['abnormal_return'].values
        
        # Train classifier
        print("   Training XGBoost Classifier...")
        clf_model, clf_results = self.train_xgboost_classifier(
            X_train, y_train_clf, X_test, y_test_clf
        )
        
        print(f"      ROC-AUC:  {clf_results['roc_auc']:.1%}  ← PRIMARY METRIC")
        print(f"      Accuracy: {clf_results['accuracy']:.1%}")
        print(f"      F1:       {clf_results['f1']:.1%}")
        print()
        
        # Train regressor
        print("   Training XGBoost Regressor...")
        reg_model, reg_results = self.train_xgboost_regressor(
            X_train, y_train_reg, X_test, y_test_reg
        )
        
        print(f"      RMSE:      {reg_results['rmse']:.4f}")
        print(f"      R²:        {reg_results['r2']:.4f}")
        print(f"      Direction: {reg_results['direction']:.1%}")
        print()
        
        # Store results
        result = {
            'feature_set': feature_key,
            'n_features': len(cols),
            'classifier': clf_results,
            'regressor': reg_results
        }
        
        return result
    
    def run(self):
        """Execute complete XGBoost workflow"""
        # Load data
        train_df, test_df, feature_info = self.load_data()
        feature_sets = self.get_feature_sets(feature_info, train_df)
        
        print("📊 Feature Sets:")
        for name, cols in feature_sets.items():
            print(f"   {name}: {len(cols)} features")
        print()
        
        # Run experiments
        experiments = [
            ('7A: XGBOOST — FINANCIAL FEATURES ONLY', 'financial_only'),
            ('7B: XGBOOST — SENTIMENT FEATURES ONLY', 'sentiment_only'),
            ('7C: XGBOOST — COMBINED FEATURES', 'combined'),
            ('7D: XGBOOST — FULL FEATURES (with FinBERT)', 'full_with_finbert')
        ]
        
        for label, key in experiments:
            result = self.run_experiment(label, key, train_df, test_df, feature_sets)
            self.all_results[key] = result
        
        # Summary
        self.display_summary()
        self.save_results()
        
        return self.all_results
    
    def display_summary(self):
        """Display results summary"""
        print("="*60)
        print("📊 XGBOOST RESULTS SUMMARY")
        print("="*60)
        print()
        
        print("CLASSIFICATION (ROC-AUC):")
        print(f"{'Feature Set':<25} {'ROC-AUC':<12} {'Accuracy':<12} {'F1'}")
        print("-"*65)
        
        for key, result in self.all_results.items():
            name = key.replace('_', ' ').title()
            clf = result['classifier']
            print(f"{name:<25} {clf['roc_auc']:<12.1%} {clf['accuracy']:<12.1%} {clf['f1']:.1%}")
        
        print()
        print("REGRESSION:")
        print(f"{'Feature Set':<25} {'RMSE':<12} {'R²':<12} {'Direction'}")
        print("-"*60)
        
        for key, result in self.all_results.items():
            name = key.replace('_', ' ').title()
            reg = result['regressor']
            print(f"{name:<25} {reg['rmse']:<12.4f} {reg['r2']:<12.4f} {reg['direction']:.1%}")
        
        print()
        
        # Best model
        best_key = max(self.all_results.items(), 
                      key=lambda x: x[1]['classifier']['roc_auc'])[0]
        best_auc = self.all_results[best_key]['classifier']['roc_auc']
        
        print(f"🏆 BEST MODEL: {best_key.replace('_', ' ').title()}")
        print(f"   ROC-AUC: {best_auc:.1%}")
        print()
    
    def save_results(self):
        """Save results to JSON"""
        print("="*60)
        print("💾 SAVING RESULTS")
        print("="*60)
        print()
        
        # Prepare results for JSON
        json_results = {}
        for key, result in self.all_results.items():
            json_results[key] = {
                'feature_set': result['feature_set'],
                'n_features': result['n_features'],
                'classifier': {k: v for k, v in result['classifier'].items() 
                             if k != 'confusion_matrix'},
                'regressor': result['regressor']
            }
        
        # Save
        output_file = self.results_dir / 'phase7_xgboost_results.json'
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved: {output_file.name}")
        print()
        
        print("="*60)
        print("✅ PHASE 7 COMPLETE")
        print("="*60)
        print()


def main():
    """Main execution"""
    trainer = XGBoostModels()
    trainer.run()
    return True


if __name__ == "__main__":
    main()