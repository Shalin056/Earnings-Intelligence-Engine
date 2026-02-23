"""
Phase 6: Baseline Models
6A: Logistic Regression (financial-only)
6B: Logistic Regression (sentiment-only)
6C: Model evaluation metrics
6D: Baseline performance documentation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR


class BaselineModels:

    def __init__(self):
        self.data_dir    = FINAL_DATA_DIR
        self.models_dir  = MODELS_DIR
        self.results_dir = RESULTS_DIR / 'baseline_models'
        self.log_file    = LOGS_DIR / 'models' / 'baseline_models.log'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.all_results = {}

    def load_data(self):
        print("📂 LOADING DATA...")
        train_df = pd.read_csv(self.data_dir / 'train_data.csv')
        test_df  = pd.read_csv(self.data_dir / 'test_data.csv')
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"✅ Features available: {len(feature_info['feature_columns'])}")
        print()
        return train_df, test_df, feature_info

    def get_feature_sets(self, feature_info, train_df):
        all_cols = set(feature_info['feature_columns']) & set(train_df.columns)
        financial = sorted([c for c in all_cols
                            if c.startswith('financial_') or c.startswith('fin_')])
        sentiment = sorted([c for c in all_cols
                            if any(c.startswith(p) for p in [
                                'sentiment_', 'mgmt_sentiment_', 'analyst_sentiment_',
                                'lm_', 'mgmt_lm_', 'analyst_lm_',
                                'word_count', 'question_count', 'lexical_diversity',
                                'mgmt_word_count', 'analyst_word_count',
                                'lm_sentiment_divergence', 'uncertainty_divergence'])])
        combined  = sorted([c for c in all_cols if not c.startswith('embedding_')])
        return {'financial_only': financial, 'sentiment_only': sentiment, 'combined': combined}

    def clean_and_scale(self, X_train, X_test):
        for X in [X_train, X_test]:
            X[np.isinf(X)] = np.nan
        medians = np.nanmedian(X_train, axis=0)
        for X in [X_train, X_test]:
            inds = np.where(np.isnan(X))
            X[inds] = np.take(medians, inds[1])
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test  = np.clip(X_test,  -1e6, 1e6)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        return X_train, X_test

    def prepare(self, train_df, test_df, cols):
        X_train = train_df[cols].values.astype(np.float64)
        X_test  = test_df[cols].values.astype(np.float64)
        return self.clean_and_scale(X_train, X_test)

    def eval_clf(self, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        return {
            'accuracy':  float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_te, y_pred, zero_division=0)),
            'f1':        float(f1_score(y_te, y_pred, zero_division=0)),
            'roc_auc':   float(roc_auc_score(y_te, y_proba)),
            'confusion_matrix': confusion_matrix(y_te, y_pred).tolist()
        }

    def eval_reg(self, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return {
            'rmse':      float(np.sqrt(mean_squared_error(y_te, y_pred))),
            'mae':       float(mean_absolute_error(y_te, y_pred)),
            'r2':        float(r2_score(y_te, y_pred)),
            'direction': float(np.mean((y_te > 0) == (y_pred > 0)))
        }

    def _run_subphase(self, label, feature_key, train_df, test_df, feature_sets, result_key):
        print("="*60)
        print(f"{label}")
        print("="*60)
        cols = feature_sets[feature_key]
        print(f"   Features: {len(cols)}")
        print()

        X_tr, X_te = self.prepare(train_df, test_df, cols)
        y_tr_clf = train_df['label_binary'].values
        y_te_clf = test_df['label_binary'].values
        y_tr_reg = train_df['abnormal_return'].values
        y_te_reg = test_df['abnormal_return'].values

        lr = self.eval_clf(LogisticRegression(C=0.1, max_iter=1000, random_state=42),
                           X_tr, y_tr_clf, X_te, y_te_clf)
        rf = self.eval_clf(RandomForestClassifier(n_estimators=100, max_depth=6,
                           random_state=42, n_jobs=-1), X_tr, y_tr_clf, X_te, y_te_clf)
        reg = self.eval_reg(Lasso(alpha=0.001, max_iter=5000),
                            X_tr, y_tr_reg, X_te, y_te_reg)

        for name, m in [('Logistic Regression', lr), ('Random Forest', rf)]:
            print(f"   {name}:")
            print(f"      ROC-AUC:  {m['roc_auc']:.1%}  ← PRIMARY METRIC")
            print(f"      Accuracy: {m['accuracy']:.1%}")
            print(f"      F1:       {m['f1']:.1%}")
            print()
        print(f"   Lasso Regression:")
        print(f"      RMSE:      {reg['rmse']:.4f}")
        print(f"      R²:        {reg['r2']:.4f}")
        print(f"      Direction: {reg['direction']:.1%}")
        print()

        result = {'feature_set': feature_key, 'n_features': len(cols),
                  'logistic_regression': lr, 'random_forest': rf, 'lasso_regression': reg}
        self.all_results[result_key] = result
        return result

    def run_6a(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6A: LOGISTIC REGRESSION — FINANCIAL FEATURES ONLY",
            'financial_only', train_df, test_df, feature_sets, '6A_financial_only')

    def run_6b(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6B: LOGISTIC REGRESSION — SENTIMENT FEATURES ONLY",
            'sentiment_only', train_df, test_df, feature_sets, '6B_sentiment_only')

    def run_6c(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6C: COMBINED MODEL + EVALUATION METRICS",
            'combined', train_df, test_df, feature_sets, '6C_combined')

    def run_6d(self):
        print("="*60)
        print("6D: BASELINE PERFORMANCE DOCUMENTATION")
        print("="*60)
        print()

        label_map = {
            '6A_financial_only': 'Financial Only',
            '6B_sentiment_only': 'Sentiment Only',
            '6C_combined':       'Combined'
        }

        print(f"{'Feature Set':<20} {'Model':<22} {'ROC-AUC':<10} {'Accuracy':<12} {'F1'}")
        print("-"*68)
        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r = self.all_results[key]
            for mname, mkey in [('Logistic Reg', 'logistic_regression'),
                                 ('Random Forest', 'random_forest')]:
                m = r[mkey]
                print(f"{label:<20} {mname:<22} {m['roc_auc']:<10.1%} "
                      f"{m['accuracy']:<12.1%} {m['f1']:.1%}")
        print()

        print(f"{'Feature Set':<20} {'RMSE':<12} {'R²':<12} {'Direction'}")
        print("-"*52)
        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r = self.all_results[key]['lasso_regression']
            print(f"{label:<20} {r['rmse']:<12.4f} {r['r2']:<12.4f} {r['direction']:.1%}")
        print()

        print("📊 THRESHOLD CHECK (ROC-AUC ≥ 0.55):")
        doc = {'phase': '6D', 'timestamp': datetime.now().isoformat(),
               'target_metric': 'ROC-AUC', 'target_threshold': 0.55, 'results': {}}
        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r = self.all_results[key]
            best = max(r['logistic_regression']['roc_auc'], r['random_forest']['roc_auc'])
            status = "✅ MEETS" if best >= 0.55 else "❌ BELOW"
            print(f"   {label:<20} Best AUC: {best:.1%}  {status} threshold")
            doc['results'][key] = {
                'feature_set': r['feature_set'],
                'n_features': r['n_features'],
                'best_roc_auc': best,
                'meets_threshold': best >= 0.55,
                'logistic_regression': {k: v for k, v in r['logistic_regression'].items()
                                        if k != 'confusion_matrix'},
                'random_forest': {k: v for k, v in r['random_forest'].items()
                                  if k != 'confusion_matrix'},
                'lasso_regression': r['lasso_regression']
            }
        print()

        doc_path = self.results_dir / 'phase6_documentation.json'
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"✅ Documentation saved: {doc_path.name}")
        print()

    def run(self):
        print("="*60)
        print("PHASE 6: BASELINE MODELS")
        print("="*60)
        print()
        train_df, test_df, feature_info = self.load_data()
        feature_sets = self.get_feature_sets(feature_info, train_df)
        print(f"📊 Feature sets:")
        print(f"   Financial only: {len(feature_sets['financial_only'])} features")
        print(f"   Sentiment only: {len(feature_sets['sentiment_only'])} features")
        print(f"   Combined:       {len(feature_sets['combined'])} features")
        print()
        self.run_6a(train_df, test_df, feature_sets)
        self.run_6b(train_df, test_df, feature_sets)
        self.run_6c(train_df, test_df, feature_sets)
        self.run_6d()
        print("="*60)
        print("✅ PHASE 6 COMPLETE")
        print("="*60)
        print()
        return self.all_results


def main():
    BaselineModels().run()

if __name__ == "__main__":
    main()