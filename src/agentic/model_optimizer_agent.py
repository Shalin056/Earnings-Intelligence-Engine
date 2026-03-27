"""
Phase 10: Agent-Enhanced Model Optimization
Autonomous AI agent that optimizes models through hyperparameter tuning and ensembling
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
from scipy.stats import uniform, randint

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR


class AgenticModelOptimizer:
    """
    Autonomous AI agent for model optimization through hyperparameter tuning and ensembling
    """
    
    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.results_dir = RESULTS_DIR / 'agentic_models'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'agentic' / 'model_optimization.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.optimization_history = {
            'baseline_models': {},
            'optimized_models': {},
            'ensemble_results': {},
            'best_configuration': {}
        }
        
        self.best_models = {}
    
    def load_data(self):
        """Load preprocessed data"""
        print("="*60)
        print("PHASE 10: AGENT-ENHANCED MODEL OPTIMIZATION")
        print("="*60)
        print()
        print("🤖 AI OPTIMIZATION AGENT INITIALIZING...")
        print()
        print("📂 Loading data...")
        
        # Load full dataset
        df = pd.read_csv(self.data_dir / 'final_dataset.csv')
        
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        print(f"✅ Loaded dataset: {len(df)} records")
        print(f"✅ Features available: {len(feature_info['feature_columns'])}")
        print()
        
        return df, feature_info
    
    def prepare_data(self, df, feature_cols):
        """Prepare clean data for modeling"""
        
        # Filter out leakage columns
        LEAK_PATTERNS = [
            'price_after', 'price_before', 'stock_return',
            'abnormal_return', 'label_binary', 'label_median',
            'label_tertile', 'sp500_return', 'alignment_timestamp',
            'is_temporally_valid', 'ticker', 'company_name',
            'quarter', 'transcript_date', 'financial_date',
            'market_date', 'DateDiff', 'date_diff'
        ]
        
        clean_features = [
            f for f in feature_cols
            if not any(pattern in f for pattern in LEAK_PATTERNS)
            and f in df.columns
        ]
        
        # Get features and target
        X = df[clean_features].values.astype(np.float64)
        y = df['label_binary'].values
        
        # Clean data
        X[np.isinf(X)] = np.nan
        medians = np.nanmedian(X, axis=0)
        
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
        
        X = np.clip(X, -1e6, 1e6)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, clean_features
    
    def get_feature_sets(self, feature_info, df):
        """Get different feature sets for optimization"""
        all_cols = set(feature_info['feature_columns']) & set(df.columns)
        
        financial = [c for c in all_cols if c.startswith('financial_')]
        sentiment = [c for c in all_cols if 'sentiment' in c.lower() or c.startswith('lm_')]
        combined = [c for c in all_cols if not c.startswith('embedding_')]
        
        return {
            'financial_only': financial,
            'sentiment_only': sentiment,
            'combined': combined
        }
    
    def run_10a_baseline_models(self, X, y):
        """
        Phase 10A: Establish baseline with default hyperparameters
        """
        print("="*60)
        print("10A: BASELINE MODEL PERFORMANCE")
        print("="*60)
        print()
        print("📊 Training baseline models (default hyperparameters)...")
        print()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Baseline models with default params
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        baseline_results = {}
        
        for name, model in models.items():
            print(f"Testing: {name}")
            
            fold_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                auc = roc_auc_score(y_test, y_proba)
                fold_scores.append(auc)
            
            baseline_results[name] = {
                'mean_auc': np.mean(fold_scores),
                'std_auc': np.std(fold_scores),
                'fold_scores': fold_scores
            }
            
            print(f"  AUC: {baseline_results[name]['mean_auc']:.3f} (±{baseline_results[name]['std_auc']:.3f})")
            print()
        
        self.optimization_history['baseline_models'] = baseline_results
        
        print("✅ Baseline established")
        print()
        
        return baseline_results
    
    def run_10b_hyperparameter_optimization(self, X, y):
        """
        Phase 10B: Agent-driven hyperparameter optimization
        """
        print("="*60)
        print("10B: AGENT HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        print()
        print("🤖 Agent is searching optimal hyperparameters...")
        print()
        
        tscv = TimeSeriesSplit(n_splits=3)
        optimized_results = {}
        
        # XGBoost hyperparameter space
        print("🔍 Optimizing XGBoost...")
        xgb_param_space = {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(100, 300),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 7),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(1, 3)
        }
        
        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=42, n_jobs=-1),
            xgb_param_space,
            n_iter=30,
            cv=tscv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        xgb_search.fit(X, y)
        
        optimized_results['XGBoost'] = {
            'best_params': xgb_search.best_params_,
            'best_score': xgb_search.best_score_,
            'best_model': xgb_search.best_estimator_
        }
        
        print(f"  Best AUC: {xgb_search.best_score_:.3f}")
        print(f"  Best params: {xgb_search.best_params_}")
        print()
        
        # Random Forest hyperparameter space
        print("🔍 Optimizing Random Forest...")
        rf_param_space = {
            'n_estimators': randint(100, 300),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_param_space,
            n_iter=20,
            cv=tscv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        rf_search.fit(X, y)
        
        optimized_results['Random Forest'] = {
            'best_params': rf_search.best_params_,
            'best_score': rf_search.best_score_,
            'best_model': rf_search.best_estimator_
        }
        
        print(f"  Best AUC: {rf_search.best_score_:.3f}")
        print(f"  Best params: {rf_search.best_params_}")
        print()
        
        # Logistic Regression hyperparameter space
        print("🔍 Optimizing Logistic Regression...")
        lr_param_space = {
            'C': uniform(0.001, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        lr_search = RandomizedSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            lr_param_space,
            n_iter=20,
            cv=tscv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        lr_search.fit(X, y)
        
        optimized_results['Logistic Regression'] = {
            'best_params': lr_search.best_params_,
            'best_score': lr_search.best_score_,
            'best_model': lr_search.best_estimator_
        }
        
        print(f"  Best AUC: {lr_search.best_score_:.3f}")
        print(f"  Best params: {lr_search.best_params_}")
        print()
        
        self.optimization_history['optimized_models'] = {
            k: {
                'best_params': v['best_params'],
                'best_score': float(v['best_score'])
            }
            for k, v in optimized_results.items()
        }
        
        self.best_models = optimized_results
        
        print("✅ Hyperparameter optimization complete")
        print()
        
        return optimized_results
    
    def run_10c_ensemble_construction(self, X, y, optimized_models):
        """
        Phase 10C: Agent constructs intelligent ensembles
        """
        print("="*60)
        print("10C: AGENT ENSEMBLE CONSTRUCTION")
        print("="*60)
        print()
        print("🤖 Agent is building ensemble models...")
        print()
        
        tscv = TimeSeriesSplit(n_splits=3)
        ensemble_results = {}
        
        # Voting Ensemble (Soft Voting)
        print("🔨 Building Voting Ensemble...")
        
        estimators = [
            ('lr', optimized_models['Logistic Regression']['best_model']),
            ('rf', optimized_models['Random Forest']['best_model']),
            ('xgb', optimized_models['XGBoost']['best_model'])
        ]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Cross-validate ensemble
        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            voting_clf.fit(X_train, y_train)
            y_proba = voting_clf.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_proba)
            fold_scores.append(auc)
        
        ensemble_results['Voting Ensemble'] = {
            'mean_auc': np.mean(fold_scores),
            'std_auc': np.std(fold_scores),
            'fold_scores': fold_scores,
            'model': voting_clf
        }
        
        print(f"  AUC: {ensemble_results['Voting Ensemble']['mean_auc']:.3f} "
              f"(±{ensemble_results['Voting Ensemble']['std_auc']:.3f})")
        print()
        
        # Weighted Ensemble (agent learns optimal weights)
        print("🔨 Building Weighted Ensemble...")
        
        # Test different weight combinations
        best_weights = None
        best_weighted_score = 0
        
        weight_combinations = [
            [0.2, 0.3, 0.5],  # XGBoost-heavy
            [0.3, 0.3, 0.4],  # Balanced
            [0.4, 0.2, 0.4],  # LR + XGBoost
            [0.25, 0.25, 0.5], # XGBoost-focused
        ]
        
        for weights in weight_combinations:
            weighted_clf = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
                n_jobs=-1
            )
            
            fold_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                weighted_clf.fit(X_train, y_train)
                y_proba = weighted_clf.predict_proba(X_test)[:, 1]
                
                auc = roc_auc_score(y_test, y_proba)
                fold_scores.append(auc)
            
            mean_score = np.mean(fold_scores)
            
            if mean_score > best_weighted_score:
                best_weighted_score = mean_score
                best_weights = weights
                ensemble_results['Weighted Ensemble'] = {
                    'mean_auc': mean_score,
                    'std_auc': np.std(fold_scores),
                    'fold_scores': fold_scores,
                    'weights': weights,
                    'model': weighted_clf
                }
        
        print(f"  Best weights: {best_weights}")
        print(f"  AUC: {ensemble_results['Weighted Ensemble']['mean_auc']:.3f} "
              f"(±{ensemble_results['Weighted Ensemble']['std_auc']:.3f})")
        print()
        
        self.optimization_history['ensemble_results'] = {
            k: {
                'mean_auc': float(v['mean_auc']),
                'std_auc': float(v['std_auc']),
                'weights': v.get('weights', None)
            }
            for k, v in ensemble_results.items()
        }
        
        print("✅ Ensemble construction complete")
        print()
        
        return ensemble_results
    
    def run_10d_comparative_analysis(self, baseline_results, optimized_results, ensemble_results):
        """
        Phase 10D: Agent analyzes and selects best configuration
        """
        print("="*60)
        print("10D: AGENT COMPARATIVE ANALYSIS")
        print("="*60)
        print()
        print("📊 Agent is comparing all configurations...")
        print()
        
        # Compile all results
        all_results = {}
        
        # Baseline
        for name, result in baseline_results.items():
            all_results[f"Baseline {name}"] = result['mean_auc']
        
        # Optimized
        for name, result in optimized_results.items():
            all_results[f"Optimized {name}"] = result['best_score']
        
        # Ensemble
        for name, result in ensemble_results.items():
            all_results[name] = result['mean_auc']
        
        # Find best
        best_config = max(all_results.items(), key=lambda x: x[1])
        
        print("🏆 PERFORMANCE COMPARISON:")
        print()
        print(f"{'Configuration':<35} {'AUC':<10} {'Improvement'}")
        print("-" * 60)
        
        baseline_best = max(baseline_results.values(), key=lambda x: x['mean_auc'])['mean_auc']
        
        for config, auc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
            improvement = auc - baseline_best
            improvement_pct = (improvement / baseline_best) * 100
            
            marker = "🥇" if config == best_config[0] else "  "
            
            print(f"{marker} {config:<35} {auc:.3f}      {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        print()
        print(f"🏆 BEST CONFIGURATION: {best_config[0]}")
        print(f"   Final AUC: {best_config[1]:.3f}")
        print(f"   Improvement over baseline: +{(best_config[1] - baseline_best):.3f} "
              f"({((best_config[1] - baseline_best) / baseline_best * 100):+.1f}%)")
        print()
        
        # Store best configuration
        self.optimization_history['best_configuration'] = {
            'name': best_config[0],
            'auc': float(best_config[1]),
            'baseline_auc': float(baseline_best),
            'improvement': float(best_config[1] - baseline_best),
            'improvement_pct': float((best_config[1] - baseline_best) / baseline_best * 100)
        }
        
        return best_config
    
    def save_results(self):
        """Save optimization results"""
        print("="*60)
        print("💾 SAVING OPTIMIZATION RESULTS")
        print("="*60)
        print()
        
        # Save optimization history
        history_path = self.results_dir / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        
        print(f"✅ Optimization history saved: {history_path.name}")
        
        # Save best models (metadata only, not actual models)
        best_models_info = {
            name: {
                'best_params': info['best_params'],
                'best_score': float(info['best_score'])
            }
            for name, info in self.best_models.items()
        }
        
        models_path = self.results_dir / 'best_models_config.json'
        with open(models_path, 'w') as f:
            json.dump(best_models_info, f, indent=2)
        
        print(f"✅ Best model configs saved: {models_path.name}")
        
        # Generate report
        report = {
            'phase': 'Phase 10: Agent-Enhanced Model Optimization',
            'timestamp': datetime.now().isoformat(),
            'optimization_strategies': [
                'Hyperparameter Optimization (RandomizedSearchCV)',
                'Ensemble Construction (Voting)',
                'Weight Optimization (Grid Search)',
                'Comparative Analysis'
            ],
            'models_optimized': list(self.best_models.keys()),
            'ensembles_created': list(self.optimization_history['ensemble_results'].keys()),
            'best_configuration': self.optimization_history['best_configuration']
        }
        
        report_path = self.results_dir / 'phase10_optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Optimization report saved: {report_path.name}")
        print()
    
    def run(self):
        """Execute complete model optimization workflow"""
        
        # Load data
        df, feature_info = self.load_data()
        
        # Get feature sets
        feature_sets = self.get_feature_sets(feature_info, df)
        
        # Use combined features (best from Phase 6-9)
        print("📊 Using 'combined' feature set (financial + sentiment + text)")
        print()
        
        # Prepare data
        X, y, clean_features = self.prepare_data(df, feature_sets['combined'])
        
        print(f"✅ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        print()
        
        # 10A: Baseline
        baseline_results = self.run_10a_baseline_models(X, y)
        
        # 10B: Hyperparameter optimization
        optimized_results = self.run_10b_hyperparameter_optimization(X, y)
        
        # 10C: Ensemble construction
        ensemble_results = self.run_10c_ensemble_construction(X, y, optimized_results)
        
        # 10D: Comparative analysis
        best_config = self.run_10d_comparative_analysis(
            baseline_results, optimized_results, ensemble_results
        )
        
        # Save results
        self.save_results()
        
        print("="*60)
        print("✅ PHASE 10 COMPLETE")
        print("="*60)
        print()
        print(f"📁 Results saved to: {self.results_dir}/")
        print()
        print("🎯 Agent successfully optimized models!")
        print(f"   Best configuration: {best_config[0]}")
        print(f"   Final AUC: {best_config[1]:.3f}")
        print()
        print("✅ Ready for Phase 11: Comparative Analysis")
        print()
        
        return True


def main():
    """Main execution"""
    agent = AgenticModelOptimizer()
    agent.run()
    return True


if __name__ == "__main__":
    main()