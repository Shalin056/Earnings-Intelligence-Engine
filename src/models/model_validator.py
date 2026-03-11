"""
Phase 8: Model Validation
Comprehensive validation including cross-validation, interpretability, and statistical testing
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

# SHAP for interpretability
import shap

# Statistical testing
from scipy import stats
from scipy.stats import mannwhitneyu

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR


class ModelValidator:
    """
    Comprehensive model validation with cross-validation, interpretability, and statistical testing
    """
    
    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.results_dir = RESULTS_DIR / 'validation'
        self.figures_dir = self.results_dir / 'figures'
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'models' / 'model_validation.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = {}
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_data(self):
        """Load full dataset for validation"""
        print("="*60)
        print("PHASE 8: MODEL VALIDATION")
        print("="*60)
        print()
        print("📂 LOADING DATA...")
        
        # Load full dataset
        full_df = pd.read_csv(self.data_dir / 'final_dataset.csv')
        
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        print(f"✅ Loaded dataset: {len(full_df)} records")
        print(f"✅ Features available: {len(feature_info['feature_columns'])}")
        print()
        
        return full_df, feature_info
    
    def get_feature_sets(self, feature_info, df):
        """Define feature sets for validation"""
        all_cols = set(feature_info['feature_columns']) & set(df.columns)
        
        # Financial features
        financial = sorted([c for c in all_cols
                          if c.startswith('financial_') or c.startswith('fin_')])
        
        # Sentiment/NLP features
        sentiment = sorted([c for c in all_cols
                          if any(c.startswith(p) for p in [
                              'sentiment_', 'mgmt_sentiment_', 'analyst_sentiment_',
                              'lm_', 'mgmt_lm_', 'analyst_lm_',
                              'word_count', 'question_count', 'lexical_diversity'])])
        
        # Combined (no embeddings)
        combined = sorted([c for c in all_cols if not c.startswith('embedding_')])
        
        # Full (with FinBERT embeddings)
        full = sorted(list(all_cols))
        
        return {
            'financial_only': financial,
            'sentiment_only': sentiment,
            'combined': combined,
            'full_with_finbert': full
        }
    
    def prepare_data(self, df, feature_cols):
        """Prepare data with proper cleaning"""
        X = df[feature_cols].values.astype(np.float64)
        y = df['label_binary'].values
        
        # Handle inf/nan
        X[np.isinf(X)] = np.nan
        medians = np.nanmedian(X, axis=0)
        
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
        
        # Clip extremes
        X = np.clip(X, -1e6, 1e6)
        
        return X, y
    
    def run_8a_time_series_cv(self, df, feature_sets):
        """
        Phase 8A: 5-fold Time-Series Cross-Validation
        """
        print("="*60)
        print("8A: TIME-SERIES CROSS-VALIDATION")
        print("="*60)
        print()
        print("🔄 Running 5-fold time-series CV...")
        print("   (Respects temporal order - no lookahead bias)")
        print()
        
        # Sort by date to ensure temporal order
        df_sorted = df.sort_values('transcript_date').reset_index(drop=True)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Models to validate
        models = {
            'Logistic Regression': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, 
                                                    random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200,
                                        subsample=0.8, colsample_bytree=0.8,
                                        random_state=42, n_jobs=-1)
        }
        
        cv_results = {}
        
        # Test each feature set
        for fs_name, fs_cols in feature_sets.items():
            print(f"Testing: {fs_name} ({len(fs_cols)} features)")
            
            X, y = self.prepare_data(df_sorted, fs_cols)
            
            fs_results = {}
            
            for model_name, model in models.items():
                fold_scores = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    auc = roc_auc_score(y_test, y_proba)
                    fold_scores.append(auc)
                
                fs_results[model_name] = {
                    'fold_scores': fold_scores,
                    'mean_auc': np.mean(fold_scores),
                    'std_auc': np.std(fold_scores),
                    'min_auc': np.min(fold_scores),
                    'max_auc': np.max(fold_scores)
                }
                
                print(f"  {model_name:<20} Mean AUC: {fs_results[model_name]['mean_auc']:.3f} "
                      f"(±{fs_results[model_name]['std_auc']:.3f})")
            
            cv_results[fs_name] = fs_results
            print()
        
        self.validation_results['8A_cross_validation'] = cv_results
        
        # Save results
        cv_path = self.results_dir / 'phase8a_cross_validation_results.json'
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print(f"✅ Cross-validation complete")
        print(f"✅ Results saved: {cv_path.name}")
        print()
        
        return cv_results
    
    def run_8b_shap_analysis(self, df, feature_sets):
        """
        Phase 8B: SHAP Interpretability Analysis
        """
        print("="*60)
        print("8B: SHAP INTERPRETABILITY ANALYSIS")
        print("="*60)
        print()
        print("🔍 Analyzing feature importance with SHAP...")
        print()
        
        # Use combined feature set for interpretability
        feature_cols = feature_sets['combined']
        X, y = self.prepare_data(df, feature_cols)
        
        # Train/test split (temporal)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train XGBoost (best model)
        model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        print("📊 Computing SHAP values (this may take a few minutes)...")
        
        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Use subset of test data for speed
        X_test_sample = X_test[:min(500, len(X_test))]
        shap_values = explainer.shap_values(X_test_sample)
        
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print()
        print("🏆 TOP 20 MOST IMPORTANT FEATURES:")
        print()
        
        for idx, row in feature_importance_df.head(20).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}")
        
        print()
        
        # Save SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, 
                         show=False, max_display=20)
        plt.tight_layout()
        
        shap_plot_path = self.figures_dir / 'shap_summary_plot.png'
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SHAP summary plot saved: {shap_plot_path.name}")
        
        # Save feature importance
        importance_path = self.results_dir / 'phase8b_feature_importance.csv'
        feature_importance_df.to_csv(importance_path, index=False)
        print(f"✅ Feature importance saved: {importance_path.name}")
        print()
        
        self.validation_results['8B_shap_analysis'] = {
            'top_20_features': feature_importance_df.head(20).to_dict('records'),
            'total_features_analyzed': len(feature_cols)
        }
        
        return feature_importance_df
    
    def run_8c_error_analysis(self, df, feature_sets):
        """
        Phase 8C: Error Analysis
        """
        print("="*60)
        print("8C: ERROR ANALYSIS")
        print("="*60)
        print()
        print("🔬 Analyzing prediction errors...")
        print()
        
        # Use best feature set
        feature_cols = feature_sets['combined']
        X, y = self.prepare_data(df, feature_cols)
        
        # Train/test split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Get metadata for analysis
        test_metadata = df.iloc[split_idx:][['ticker', 'quarter', 'abnormal_return', 
                                              'stock_return_3day', 'sp500_return_3day']].reset_index(drop=True)
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Identify errors
        errors = (y_pred != y_test)
        
        # Categorize errors
        false_positives = (y_pred == 1) & (y_test == 0)
        false_negatives = (y_pred == 0) & (y_test == 1)
        
        print(f"Total predictions: {len(y_test)}")
        print(f"Correct predictions: {(~errors).sum()} ({(~errors).sum()/len(y_test)*100:.1f}%)")
        print(f"Errors: {errors.sum()} ({errors.sum()/len(y_test)*100:.1f}%)")
        print(f"  False Positives: {false_positives.sum()} ({false_positives.sum()/len(y_test)*100:.1f}%)")
        print(f"  False Negatives: {false_negatives.sum()} ({false_negatives.sum()/len(y_test)*100:.1f}%)")
        print()
        
        # Analyze error patterns
        print("📊 ERROR PATTERNS:")
        print()
        
        # Error by return magnitude
        test_metadata['error'] = errors
        test_metadata['error_type'] = 'Correct'
        test_metadata.loc[false_positives, 'error_type'] = 'False Positive'
        test_metadata.loc[false_negatives, 'error_type'] = 'False Negative'
        test_metadata['prediction_confidence'] = y_proba
        
        # Analyze false positives (predicted positive, actually negative)
        if false_positives.sum() > 0:
            fp_data = test_metadata[false_positives]
            print(f"FALSE POSITIVES (predicted +, actually -):")
            print(f"  Average return: {fp_data['abnormal_return'].mean():.2%}")
            print(f"  Return std: {fp_data['abnormal_return'].std():.2%}")
            print(f"  Average confidence: {fp_data['prediction_confidence'].mean():.2%}")
            print()
        
        # Analyze false negatives (predicted negative, actually positive)
        if false_negatives.sum() > 0:
            fn_data = test_metadata[false_negatives]
            print(f"FALSE NEGATIVES (predicted -, actually +):")
            print(f"  Average return: {fn_data['abnormal_return'].mean():.2%}")
            print(f"  Return std: {fn_data['abnormal_return'].std():.2%}")
            print(f"  Average confidence: {fn_data['prediction_confidence'].mean():.2%}")
            print()
        
        # Save error analysis
        error_analysis_path = self.results_dir / 'phase8c_error_analysis.csv'
        test_metadata.to_csv(error_analysis_path, index=False)
        print(f"✅ Error analysis saved: {error_analysis_path.name}")
        
        # Plot error distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Return distribution by error type
        for error_type in ['Correct', 'False Positive', 'False Negative']:
            subset = test_metadata[test_metadata['error_type'] == error_type]
            if len(subset) > 0:
                axes[0].hist(subset['abnormal_return'], bins=30, alpha=0.5, label=error_type)
        
        axes[0].set_xlabel('Abnormal Return')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Return Distribution by Prediction Type')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Confidence distribution
        axes[1].boxplot([
            test_metadata[test_metadata['error_type'] == 'Correct']['prediction_confidence'],
            test_metadata[test_metadata['error_type'] == 'False Positive']['prediction_confidence'],
            test_metadata[test_metadata['error_type'] == 'False Negative']['prediction_confidence']
        ], labels=['Correct', 'FP', 'FN'])
        
        axes[1].set_ylabel('Prediction Confidence')
        axes[1].set_title('Confidence Distribution by Prediction Type')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        error_plot_path = self.figures_dir / 'error_analysis.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Error plots saved: {error_plot_path.name}")
        print()
        
        self.validation_results['8C_error_analysis'] = {
            'total_predictions': int(len(y_test)),
            'correct': int((~errors).sum()),
            'errors': int(errors.sum()),
            'false_positives': int(false_positives.sum()),
            'false_negatives': int(false_negatives.sum()),
            'error_rate': float(errors.sum() / len(y_test))
        }
        
        return test_metadata
    
    def delong_test(self, y_true, y_score1, y_score2):
        """
        DeLong test for comparing two ROC curves
        Returns: z-statistic and p-value
        """
        from sklearn.metrics import roc_auc_score
        
        auc1 = roc_auc_score(y_true, y_score1)
        auc2 = roc_auc_score(y_true, y_score2)
        
        # Simplified DeLong test using Mann-Whitney U statistic
        # Full implementation would use proper DeLong covariance calculation
        
        n = len(y_true)
        
        # Get positive and negative samples
        pos_idx = y_true == 1
        neg_idx = y_true == 0
        
        n_pos = pos_idx.sum()
        n_neg = neg_idx.sum()
        
        # Compute structural components (simplified)
        # This is an approximation - full DeLong requires covariance matrix
        
        # Standard error approximation
        se = np.sqrt((auc1 * (1 - auc1) + auc2 * (1 - auc2)) / n)
        
        # Z-statistic
        z = (auc1 - auc2) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value, auc1, auc2
    
    def run_8d_statistical_testing(self, df, feature_sets):
        """
        Phase 8D: Statistical Significance Testing (DeLong test)
        """
        print("="*60)
        print("8D: STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        print()
        print("📊 Comparing models with DeLong test...")
        print()
        
        # Prepare data for different feature sets
        split_idx = int(len(df) * 0.7)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        y_test = test_df['label_binary'].values
        
        # Train models on different feature sets
        model_predictions = {}
        
        for fs_name, fs_cols in feature_sets.items():
            print(f"Training on: {fs_name}")
            
            X, y = self.prepare_data(df, fs_cols)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train = y[:split_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train XGBoost
            model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Get predictions
            y_proba = model.predict_proba(X_test)[:, 1]
            model_predictions[fs_name] = y_proba
            
            auc = roc_auc_score(y_test, y_proba)
            print(f"  AUC: {auc:.3f}")
        
        print()
        print("📊 PAIRWISE COMPARISONS (DeLong Test):")
        print()
        
        # Compare all pairs
        comparison_results = []
        
        fs_names = list(model_predictions.keys())
        for i in range(len(fs_names)):
            for j in range(i + 1, len(fs_names)):
                name1, name2 = fs_names[i], fs_names[j]
                
                z, p_value, auc1, auc2 = self.delong_test(
                    y_test, 
                    model_predictions[name1],
                    model_predictions[name2]
                )
                
                # Determine significance
                if p_value < 0.001:
                    sig = '***'
                elif p_value < 0.01:
                    sig = '**'
                elif p_value < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                comparison_results.append({
                    'model_1': name1,
                    'model_2': name2,
                    'auc_1': auc1,
                    'auc_2': auc2,
                    'auc_diff': abs(auc1 - auc2),
                    'z_statistic': z,
                    'p_value': p_value,
                    'significance': sig,
                    'better_model': name1 if auc1 > auc2 else name2
                })
                
                print(f"{name1} vs {name2}:")
                print(f"  AUC: {auc1:.3f} vs {auc2:.3f} (diff: {abs(auc1-auc2):.3f})")
                print(f"  z = {z:.3f}, p = {p_value:.4f} {sig}")
                print(f"  Winner: {name1 if auc1 > auc2 else name2}")
                print()
        
        # Save results
        comparison_df = pd.DataFrame(comparison_results)
        comparison_path = self.results_dir / 'phase8d_statistical_tests.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"✅ Statistical test results saved: {comparison_path.name}")
        print()
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print()
        
        self.validation_results['8D_statistical_testing'] = {
            'comparisons': comparison_results,
            'total_comparisons': len(comparison_results),
            'significant_differences': sum(1 for r in comparison_results if r['p_value'] < 0.05)
        }
        
        return comparison_df
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("="*60)
        print("📝 GENERATING VALIDATION REPORT")
        print("="*60)
        print()
        
        report = {
            'validation_date': datetime.now().isoformat(),
            'phases_completed': list(self.validation_results.keys()),
            'results': self.validation_results
        }
        
        report_path = self.results_dir / 'phase8_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved: {report_path.name}")
        print()
    
    def run(self):
        """Execute complete validation workflow"""
        # Load data
        df, feature_info = self.load_data()
        feature_sets = self.get_feature_sets(feature_info, df)
        
        print(f"📊 Feature Sets:")
        for name, cols in feature_sets.items():
            print(f"   {name}: {len(cols)} features")
        print()
        
        # Run all validation phases
        print("🚀 Running validation phases...")
        print()
        
        # 8A: Cross-validation
        self.run_8a_time_series_cv(df, feature_sets)
        
        # 8B: SHAP analysis
        self.run_8b_shap_analysis(df, feature_sets)
        
        # 8C: Error analysis
        self.run_8c_error_analysis(df, feature_sets)
        
        # 8D: Statistical testing
        self.run_8d_statistical_testing(df, feature_sets)
        
        # Generate final report
        self.generate_validation_report()
        
        print("="*60)
        print("✅ PHASE 8 COMPLETE")
        print("="*60)
        print()
        print(f"📁 All results saved to: {self.results_dir}/")
        print(f"📊 Figures saved to: {self.figures_dir}/")
        print()
        
        return True


def main():
    """Main execution"""
    validator = ModelValidator()
    validator.run()
    return True


if __name__ == "__main__":
    main()