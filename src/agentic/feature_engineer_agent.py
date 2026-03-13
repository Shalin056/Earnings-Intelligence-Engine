"""
Phase 9: Agentic Feature Engineering
Autonomous AI agent that discovers and evaluates new features
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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
from itertools import combinations
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR


class AgenticFeatureEngineer:
    """
    Autonomous AI agent for discovering and engineering features
    """
    
    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.results_dir = RESULTS_DIR / 'agentic_features'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'agentic' / 'feature_engineering.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.discovered_features = []
        self.feature_scores = {}
        self.best_features = []
        
        # Agent learning history
        self.agent_history = {
            'iterations': [],
            'best_score_per_iteration': [],
            'features_tested': 0,
            'features_kept': 0
        }
    
    def load_data(self):
        """Load preprocessed data"""
        print("="*60)
        print("PHASE 9: AGENTIC FEATURE ENGINEERING")
        print("="*60)
        print()
        print("🤖 AI AGENT INITIALIZING...")
        print()
        print("📂 Loading data...")
        
        df = pd.read_csv(self.data_dir / 'final_dataset.csv')
        
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        print(f"✅ Loaded dataset: {len(df)} records")
        print(f"✅ Existing features: {len(feature_info['feature_columns'])}")
        print()
        
        return df, feature_info
    
    def get_base_features(self, df, feature_info):
        """Get base feature sets for engineering"""
        all_cols = set(feature_info['feature_columns']) & set(df.columns)
        
        # Categorize features
        financial = [c for c in all_cols if c.startswith('financial_')]
        sentiment = [c for c in all_cols if 'sentiment' in c.lower() or c.startswith('lm_')]
        text_stats = [c for c in all_cols if any(x in c for x in ['word_count', 'lexical', 'question'])]
        market = []
        
        return {
            'financial': financial,
            'sentiment': sentiment,
            'text_stats': text_stats,
            'market': market,
            'all_base': list(all_cols - set([c for c in all_cols if c.startswith('embedding_')]))
        }
    
    def generate_interaction_features(self, df, feature_groups):
        """
        AGENT STRATEGY 1: Generate interaction features
        Automatically discover meaningful feature interactions
        """
        print("="*60)
        print("🤖 AGENT STRATEGY 1: INTERACTION DISCOVERY")
        print("="*60)
        print()
        print("🔍 Agent is exploring feature interactions...")
        print()
        
        new_features = {}
        
        # Financial × Sentiment interactions
        print("Testing: Financial × Sentiment interactions")
        for fin_feat in tqdm(feature_groups['financial'][:10], desc="Financial features"):
            for sent_feat in feature_groups['sentiment'][:5]:
                try:
                    feat_name = f"interact_{fin_feat}×{sent_feat}"
                    new_features[feat_name] = df[fin_feat] * df[sent_feat]
                except:
                    continue
        
        print(f"  Created {len(new_features)} interaction features")
        print()
        
        # Market × Text interactions
        print("Testing: Market × Text statistics interactions")
        for market_feat in feature_groups['market']:
            for text_feat in feature_groups['text_stats']:
                try:
                    feat_name = f"interact_{market_feat}×{text_feat}"
                    new_features[feat_name] = df[market_feat] * df[text_feat]
                except:
                    continue
        
        print(f"  Total interaction features: {len(new_features)}")
        print()
        
        return new_features
    
    def generate_polynomial_features(self, df, feature_groups):
        """
        AGENT STRATEGY 2: Generate polynomial features
        Discover non-linear relationships
        """
        print("="*60)
        print("🤖 AGENT STRATEGY 2: POLYNOMIAL DISCOVERY")
        print("="*60)
        print()
        print("🔍 Agent is exploring polynomial transformations...")
        print()
        
        new_features = {}
        
        # Top financial features (squared and cubed)
        top_financial = feature_groups['financial'][:15]
        
        print("Testing: Polynomial features (x², x³, √x)")
        for feat in tqdm(top_financial, desc="Polynomial"):
            try:
                # Square
                new_features[f"poly_{feat}²"] = df[feat] ** 2
                
                # Cube
                # new_features[f"poly_{feat}³"] = df[feat] ** 3
                
                # Square root (for positive values)
                if (df[feat] >= 0).all():
                    new_features[f"poly_{feat}^0.5"] = np.sqrt(df[feat])
                
                # Log (for positive values)
                if (df[feat] > 0).all():
                    new_features[f"poly_log_{feat}"] = np.log(df[feat] + 1)
                
            except:
                continue
        
        print(f"  Created {len(new_features)} polynomial features")
        print()
        
        return new_features
    
    def generate_ratio_features(self, df, feature_groups):
        """
        AGENT STRATEGY 3: Generate ratio features
        Discover relative relationships
        """
        print("="*60)
        print("🤖 AGENT STRATEGY 3: RATIO DISCOVERY")
        print("="*60)
        print()
        print("🔍 Agent is exploring ratio relationships...")
        print()
        
        new_features = {}
        
        # Financial ratios
        financial = feature_groups['financial'][:10]
        
        print("Testing: Financial ratio features")
        for i, feat1 in enumerate(financial):
            for feat2 in financial[i+1:i+4]:  # Limit combinations
                try:
                    # Avoid division by zero
                    denominator = df[feat2].replace(0, np.nan)
                    feat_name = f"ratio_{feat1}/{feat2}"
                    new_features[feat_name] = df[feat1] / denominator
                except:
                    continue
        
        # Sentiment ratios
        sentiment = feature_groups['sentiment'][:8]
        
        print("Testing: Sentiment ratio features")
        for i, feat1 in enumerate(sentiment):
            for feat2 in sentiment[i+1:i+3]:
                try:
                    denominator = df[feat2].replace(0, np.nan)
                    feat_name = f"ratio_{feat1}/{feat2}"
                    new_features[feat_name] = df[feat1] / denominator
                except:
                    continue
        
        print(f"  Created {len(new_features)} ratio features")
        print()
        
        return new_features
    
    def generate_aggregation_features(self, df, feature_groups):
        """
        AGENT STRATEGY 4: Generate aggregation features
        Discover combined signals
        """
        print("="*60)
        print("🤖 AGENT STRATEGY 4: AGGREGATION DISCOVERY")
        print("="*60)
        print()
        print("🔍 Agent is exploring feature aggregations...")
        print()
        
        new_features = {}
        
        # Financial aggregations
        financial = feature_groups['financial']
        if len(financial) > 0:
            df_financial = df[financial].fillna(0)
            
            new_features['agg_financial_mean'] = df_financial.mean(axis=1)
            new_features['agg_financial_std'] = df_financial.std(axis=1)
            new_features['agg_financial_max'] = df_financial.max(axis=1)
            new_features['agg_financial_min'] = df_financial.min(axis=1)
        
        # Sentiment aggregations
        sentiment = [c for c in feature_groups['sentiment'] if 'score' in c]
        if len(sentiment) > 0:
            df_sentiment = df[sentiment].fillna(0)
            
            new_features['agg_sentiment_mean'] = df_sentiment.mean(axis=1)
            new_features['agg_sentiment_std'] = df_sentiment.std(axis=1)
            new_features['agg_sentiment_range'] = df_sentiment.max(axis=1) - df_sentiment.min(axis=1)
        
        print(f"  Created {len(new_features)} aggregation features")
        print()
        
        return new_features
    
    def evaluate_features(self, df, new_features, target):
        """
        AGENT EVALUATION: Score each discovered feature
        """
        print("="*60)
        print("🤖 AGENT EVALUATION: SCORING FEATURES")
        print("="*60)
        print()
        print("📊 Agent is evaluating discovered features...")
        print()
        
        # Clean features
        new_features_df = pd.DataFrame(new_features)
        
        # Replace inf with nan
        new_features_df = new_features_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill nan with median
        for col in new_features_df.columns:
            if new_features_df[col].isna().sum() > 0:
                median_val = new_features_df[col].median()
                new_features_df[col] = new_features_df[col].fillna(median_val)
        
        # Calculate mutual information scores
        print("Computing mutual information scores...")
        
        try:
            mi_scores = mutual_info_classif(
                new_features_df, 
                target, 
                discrete_features=False,
                random_state=42,
                n_neighbors=5
            )
            
            # Store scores
            feature_scores = dict(zip(new_features_df.columns, mi_scores))
            
            # Sort by score
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            print()
            print("🏆 TOP 20 DISCOVERED FEATURES:")
            print()
            
            for i, (feat, score) in enumerate(sorted_features[:20], 1):
                print(f"  {i:2d}. {feat:<60} Score: {score:.4f}")
                self.discovered_features.append({
                    'rank': i,
                    'feature': feat,
                    'score': score
                })
            
            print()
            
            # Keep top features (threshold: top 50 or score > 0.01)
            threshold_score = 0.02
            top_n = 15
            
            # Remove any features built from leak columns
            LEAK_PATTERNS = [
                'DateDiff',           # catches both DateDiff and financial_DateDiff
                'financial_DateDiff', # explicit catch
                'price_after',
                'price_before',
                'stock_return',
                'alignment_timestamp',
                'label_binary',
                'label_median',
                'label_tertile',
                'abnormal_return'
            ]

            self.best_features = [
                feat for feat, score in sorted_features
                if score > threshold_score
                and not any(pattern in feat for pattern in LEAK_PATTERNS)
            ][:top_n]

            print(f"✅ Agent selected {len(self.best_features)} high-value features")
            print(f"   (Leak patterns filtered: {len(LEAK_PATTERNS)} patterns blocked)")
            print(f"   (Threshold: score > {threshold_score} OR top {top_n})")
            print()
            
            self.feature_scores = feature_scores
            
            return new_features_df[self.best_features], feature_scores
            
        except Exception as e:
            print(f"❌ Error in feature evaluation: {e}")
            return pd.DataFrame(), {}
    
    def validate_features_with_model(self, df, base_features, new_features_df, target):
        """
        AGENT VALIDATION: Test features with actual model
        """
        print("="*60)
        print("🤖 AGENT VALIDATION: MODEL TESTING")
        print("="*60)
        print()
        print("🔬 Agent is validating features with XGBoost...")
        print()
        
        # Prepare base features
        X_base = df[base_features].values
        X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale base features
        scaler_base = StandardScaler()
        X_base_scaled = scaler_base.fit_transform(X_base)
        
        # Filter leak columns from base features
        LEAK_PATTERNS = [
            'DateDiff', 'price_after', 'price_before',
            'stock_return', 'alignment_timestamp',
            'label_binary', 'label_median', 'label_tertile',
            'abnormal_return', 'is_temporally_valid',
            'ticker', 'company_name', 'quarter',
            'transcript_date', 'financial_date',
        ]
        base_features = [
            f for f in base_features
            if not any(pattern in f for pattern in LEAK_PATTERNS)
            and f in df.columns
        ]

        # Baseline model
        print("Testing baseline (human-designed features only)...")
        model_base = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            random_state=42, n_jobs=-1
        )
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores_base = cross_val_score(
            model_base, X_base_scaled, target, 
            cv=tscv, scoring='roc_auc', n_jobs=-1
        )
        
        baseline_auc = scores_base.mean()
        print(f"  Baseline AUC: {baseline_auc:.3f} (±{scores_base.std():.3f})")
        print()
        
        # Enhanced model (with agent-discovered features)
        print("Testing enhanced (baseline + agent-discovered features)...")
        
        # Combine features
        X_enhanced = np.hstack([X_base_scaled, new_features_df.values])
        
        model_enhanced = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            random_state=42, n_jobs=-1
        )
        
        scores_enhanced = cross_val_score(
            model_enhanced, X_enhanced, target,
            cv=tscv, scoring='roc_auc', n_jobs=-1
        )
        
        enhanced_auc = scores_enhanced.mean()
        print(f"  Enhanced AUC: {enhanced_auc:.3f} (±{scores_enhanced.std():.3f})")
        print()
        
        # Calculate improvement
        improvement = enhanced_auc - baseline_auc
        improvement_pct = (improvement / baseline_auc) * 100
        
        print("="*60)
        print("📊 AGENTIC FEATURE ENGINEERING RESULTS")
        print("="*60)
        print()
        print(f"Baseline (human-designed):  {baseline_auc:.3f}")
        print(f"Enhanced (+ AI-discovered): {enhanced_auc:.3f}")
        print(f"Improvement:                +{improvement:.3f} ({improvement_pct:+.1f}%)")
        print()
        
        if improvement > 0:
            print("✅ Agent successfully discovered valuable features!")
        else:
            print("⚠️  Agent-discovered features did not improve performance")
        
        print()
        
        # Store results
        self.agent_history['baseline_auc'] = float(baseline_auc)
        self.agent_history['enhanced_auc'] = float(enhanced_auc)
        self.agent_history['improvement'] = float(improvement)
        self.agent_history['improvement_pct'] = float(improvement_pct)
        
        return baseline_auc, enhanced_auc, improvement
    
    def save_discovered_features(self, new_features_df):
        """Save agent-discovered features"""
        print("="*60)
        print("💾 SAVING AGENT DISCOVERIES")
        print("="*60)
        print()
        
        # Save discovered features as CSV
        features_path = self.results_dir / 'agent_discovered_features.csv'
        new_features_df.to_csv(features_path, index=False)
        print(f"✅ Discovered features saved: {features_path.name}")
        print(f"   Count: {len(new_features_df.columns)} features")
        
        # Save feature rankings
        rankings_df = pd.DataFrame(self.discovered_features)
        rankings_path = self.results_dir / 'feature_rankings.csv'
        rankings_df.to_csv(rankings_path, index=False)
        print(f"✅ Feature rankings saved: {rankings_path.name}")
        
        # Save agent history
        history_path = self.results_dir / 'agent_learning_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.agent_history, f, indent=2)
        print(f"✅ Agent history saved: {history_path.name}")
        
        print()
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("="*60)
        print("📝 GENERATING AGENT REPORT")
        print("="*60)
        print()
        
        report = {
            'phase': 'Phase 9: Agentic Feature Engineering',
            'timestamp': datetime.now().isoformat(),
            'agent_strategies': [
                'Interaction Discovery',
                'Polynomial Discovery',
                'Ratio Discovery',
                'Aggregation Discovery'
            ],
            'features_tested': self.agent_history.get('features_tested', 0),
            'features_discovered': len(self.best_features),
            'baseline_auc': self.agent_history.get('baseline_auc', 0),
            'enhanced_auc': self.agent_history.get('enhanced_auc', 0),
            'improvement': self.agent_history.get('improvement', 0),
            'improvement_pct': self.agent_history.get('improvement_pct', 0),
            'top_10_features': self.discovered_features[:10]
        }
        
        report_path = self.results_dir / 'phase9_agentic_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Agent report saved: {report_path.name}")
        print()
        
        # Display summary
        print("🤖 AGENTIC AI SUMMARY:")
        print()
        print(f"Strategies employed: {len(report['agent_strategies'])}")
        print(f"Features discovered: {report['features_discovered']}")
        print(f"Performance improvement: {report['improvement_pct']:.1f}%")
        print()
    
    def run(self):
        """Execute complete agentic feature engineering workflow"""
        
        # Load data
        df, feature_info = self.load_data()
        
        # Get base features
        feature_groups = self.get_base_features(df, feature_info)
        
        print(f"📊 Base Feature Groups:")
        print(f"   Financial: {len(feature_groups['financial'])}")
        print(f"   Sentiment: {len(feature_groups['sentiment'])}")
        print(f"   Text Stats: {len(feature_groups['text_stats'])}")
        print(f"   Market: {len(feature_groups['market'])}")
        print()
        
        # Agent discovers features using multiple strategies
        all_new_features = {}
        
        # Strategy 1: Interactions
        interactions = self.generate_interaction_features(df, feature_groups)
        all_new_features.update(interactions)
        
        # Strategy 2: Polynomials
        polynomials = self.generate_polynomial_features(df, feature_groups)
        all_new_features.update(polynomials)
        
        # Strategy 3: Ratios
        ratios = self.generate_ratio_features(df, feature_groups)
        all_new_features.update(ratios)
        
        # Strategy 4: Aggregations
        aggregations = self.generate_aggregation_features(df, feature_groups)
        all_new_features.update(aggregations)
        
        self.agent_history['features_tested'] = len(all_new_features)
        
        print(f"🤖 AGENT DISCOVERY SUMMARY:")
        print(f"   Total features created: {len(all_new_features)}")
        print()
        
        # Get target
        target = df['label_binary'].values
        
        # Evaluate features
        new_features_df, feature_scores = self.evaluate_features(df, all_new_features, target)
        
        if len(new_features_df) > 0:
            # Validate with model
            baseline_auc, enhanced_auc, improvement = self.validate_features_with_model(
                df, feature_groups['all_base'], new_features_df, target
            )
            
            # Save results
            self.save_discovered_features(new_features_df)
            
            # Generate report
            self.generate_report()
        
        print("="*60)
        print("✅ PHASE 9 COMPLETE")
        print("="*60)
        print()
        print(f"📁 Results saved to: {self.results_dir}/")
        print()
        print("✅ Ready for Phase 10: Agent-Enhanced Modeling")
        print()
        
        return True


def main():
    """Main execution"""
    agent = AgenticFeatureEngineer()
    agent.run()
    return True


if __name__ == "__main__":
    main()