"""
Phase 3C: Feature Integration
Merges all feature sets into final training dataset
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, FEATURES_DIR, FINAL_DATA_DIR, LOGS_DIR

class FeatureIntegrator:
    """
    Integrates all feature sets into final training dataset
    """
    
    def __init__(self):
        self.aligned_dir = PROCESSED_DATA_DIR / 'aligned'
        self.features_dir = FEATURES_DIR
        self.output_dir = FINAL_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'features' / 'feature_integration.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.integration_stats = {
            'aligned_records': 0,
            'finbert_records': 0,
            'nlp_records': 0,
            'final_records': 0,
            'total_features': 0
        }
    
    def load_all_features(self):
        """
        Load all feature sets
        """
        print("="*60)
        print("PHASE 3C: FEATURE INTEGRATION")
        print("="*60)
        print()
        print("📂 LOADING ALL FEATURE SETS...")
        print()
        
        features = {}
        
        # 1. Load aligned data (has financial + market features)
        aligned_file = self.aligned_dir / 'aligned_data.csv'
        if not aligned_file.exists():
            print(f"❌ Aligned data not found: {aligned_file}")
            return None
        
        features['aligned'] = pd.read_csv(aligned_file)
        print(f"✅ Aligned data: {len(features['aligned'])} records, {len(features['aligned'].columns)} features")
        self.integration_stats['aligned_records'] = len(features['aligned'])
        
        # 2. Load FinBERT features
        finbert_file = self.features_dir / 'finbert_features.csv'
        if not finbert_file.exists():
            print(f"❌ FinBERT features not found: {finbert_file}")
            return None
        
        features['finbert'] = pd.read_csv(finbert_file)
        print(f"✅ FinBERT features: {len(features['finbert'])} records, {len(features['finbert'].columns)} features")
        self.integration_stats['finbert_records'] = len(features['finbert'])
        
        # 3. Load NLP features
        nlp_file = self.features_dir / 'nlp_features.csv'
        if not nlp_file.exists():
            print(f"❌ NLP features not found: {nlp_file}")
            return None
        
        features['nlp'] = pd.read_csv(nlp_file)
        print(f"✅ NLP features: {len(features['nlp'])} records, {len(features['nlp'].columns)} features")
        self.integration_stats['nlp_records'] = len(features['nlp'])
        
        print()
        return features
    
    def merge_features(self, features):
        """
        Merge all features on ticker + quarter
        """
        print("="*60)
        print("🔗 MERGING FEATURES")
        print("="*60)
        print()
        
        # Start with aligned data (has target variable)
        merged = features['aligned'].copy()
        print(f"Starting with aligned data: {len(merged)} records")
        
        # Merge FinBERT features
        print("Merging FinBERT features...")
        
        # Drop duplicate columns (ticker, quarter, date) from finbert
        finbert_merge = features['finbert'].drop(columns=['date'], errors='ignore')
        
        merged = merged.merge(
            finbert_merge,
            on=['ticker', 'quarter'],
            how='left',
            suffixes=('', '_finbert')
        )
        print(f"   After FinBERT merge: {len(merged)} records, {len(merged.columns)} features")
        
        # Merge NLP features
        print("Merging NLP features...")
        
        # Drop duplicate columns from nlp
        nlp_merge = features['nlp'].drop(columns=['date'], errors='ignore')
        
        merged = merged.merge(
            nlp_merge,
            on=['ticker', 'quarter'],
            how='left',
            suffixes=('', '_nlp')
        )
        print(f"   After NLP merge: {len(merged)} records, {len(merged.columns)} features")
        
        print()
        
        # Check for missing values
        missing = merged.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) > 0:
            print(f"⚠️  Found {len(missing_cols)} columns with missing values")
            print("   Top missing columns:")
            for col, count in missing_cols.head(10).items():
                pct = count / len(merged) * 100
                print(f"      {col}: {count} ({pct:.1f}%)")
            print()
        else:
            print("✅ No missing values detected")
            print()
        
        self.integration_stats['final_records'] = len(merged)
        self.integration_stats['total_features'] = len(merged.columns)
        
        return merged
    
    def prepare_final_dataset(self, merged):
        """
        Prepare final dataset for modeling
        """
        print("="*60)
        print("🎯 PREPARING FINAL DATASET")
        print("="*60)
        print()
        
        # Identify feature types
        metadata_cols = ['ticker', 'company_name', 'quarter', 'transcript_date', 
                        'financial_date', 'market_date_before', 'market_date_after',
                        'date', 'fiscal_year', 'fiscal_quarter', 'alignment_timestamp',
                        'is_temporally_valid', 'financial_date_diff_days']
        
        target_col = 'stock_return_3day'
        
        # Feature columns (everything except metadata and target)
        feature_cols = [col for col in merged.columns 
                       if col not in metadata_cols and col != target_col]
        
        print(f"📊 Feature Breakdown:")
        print(f"   Metadata columns: {len(metadata_cols)}")
        print(f"   Target column: 1 ({target_col})")
        print(f"   Feature columns: {len(feature_cols)}")
        print()
        
        # Categorize features
        financial_features = [col for col in feature_cols if col.startswith('financial_')]
        finbert_features = [col for col in feature_cols if col.startswith('embedding_') or 
                           'sentiment' in col.lower()]
        nlp_features = [col for col in feature_cols if col.startswith('lm_') or 
                       col in ['word_count', 'sentence_count', 'lexical_diversity', 
                               'question_count', 'avg_word_length', 'avg_sentence_length',
                               'unique_words', 'char_count']]
        market_features = [col for col in feature_cols if 'price' in col.lower() or 
                          col in ['total_management_words', 'total_analyst_words', 
                                 'management_analyst_ratio']]
        
        print("📦 Feature Categories:")
        print(f"   Financial features: {len(financial_features)}")
        print(f"   FinBERT features: {len(finbert_features)}")
        print(f"   NLP features: {len(nlp_features)}")
        print(f"   Market features: {len(market_features)}")
        print(f"   Other features: {len(feature_cols) - len(financial_features) - len(finbert_features) - len(nlp_features) - len(market_features)}")
        print()
        
        return merged, metadata_cols, target_col, feature_cols
    
    def create_train_test_split(self, df, test_size=0.3):
        """
        Create train/test split based on temporal order
        """
        print("="*60)
        print("✂️  CREATING TRAIN/TEST SPLIT")
        print("="*60)
        print()
        
        # Sort by date to maintain temporal order
        df = df.sort_values('transcript_date').reset_index(drop=True)
        
        # Split chronologically
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"📊 Split Summary:")
        print(f"   Total records: {len(df):,}")
        print(f"   Training set: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Test set: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        print()
        print(f"📅 Date Ranges:")
        print(f"   Training: {train_df['transcript_date'].min()} to {train_df['transcript_date'].max()}")
        print(f"   Test: {test_df['transcript_date'].min()} to {test_df['transcript_date'].max()}")
        print()
        
        return train_df, test_df
    
    def save_final_dataset(self, merged, train_df, test_df, metadata_cols, target_col, feature_cols):
        """
        Save final integrated dataset
        """
        print("="*60)
        print("💾 SAVING FINAL DATASET")
        print("="*60)
        print()
        
        try:
            # Save full dataset
            full_path = self.output_dir / 'final_dataset.csv'
            merged.to_csv(full_path, index=False)
            print(f"✅ Full dataset saved: {full_path}")
            print(f"   Size: {full_path.stat().st_size / (1024*1024):.2f} MB")
            print(f"   Records: {len(merged):,}")
            print(f"   Features: {len(merged.columns)}")
            
            # Save train set
            train_path = self.output_dir / 'train_data.csv'
            train_df.to_csv(train_path, index=False)
            print(f"✅ Training set saved: {train_path}")
            print(f"   Size: {train_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save test set
            test_path = self.output_dir / 'test_data.csv'
            test_df.to_csv(test_path, index=False)
            print(f"✅ Test set saved: {test_path}")
            print(f"   Size: {test_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save feature list
            feature_info = {
                'metadata_columns': metadata_cols,
                'target_column': target_col,
                'feature_columns': feature_cols,
                'total_features': len(feature_cols),
                'creation_date': datetime.now().isoformat()
            }
            
            feature_info_path = self.output_dir / 'feature_info.json'
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"✅ Feature info saved: {feature_info_path}")
            
            # Save integration statistics
            stats_path = self.output_dir / 'integration_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(self.integration_stats, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete feature integration workflow
        """
        # Load all features
        features = self.load_all_features()
        
        if features is None:
            print("❌ Feature integration failed")
            return False
        
        # Merge features
        merged = self.merge_features(features)
        
        # Prepare final dataset
        merged, metadata_cols, target_col, feature_cols = self.prepare_final_dataset(merged)
        
        # Create train/test split
        train_df, test_df = self.create_train_test_split(merged)
        
        # Save datasets
        success = self.save_final_dataset(merged, train_df, test_df, 
                                         metadata_cols, target_col, feature_cols)
        
        if success:
            print("="*60)
            print("✅ PHASE 3C COMPLETE")
            print("="*60)
            print()
            print(f"📁 Final datasets saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("="*60)
            print("🎉 PHASE 3 (FEATURE ENGINEERING) COMPLETE!")
            print("="*60)
            print()
            print(f"📊 Final Dataset Summary:")
            print(f"   Total records: {self.integration_stats['final_records']:,}")
            print(f"   Total features: {self.integration_stats['total_features']}")
            print(f"   Training samples: {len(train_df):,}")
            print(f"   Test samples: {len(test_df):,}")
            print()
            print("✅ Ready for Phase 4: Baseline Modeling")
            print()
        
        return success

def main():
    """Main execution"""
    integrator = FeatureIntegrator()
    success = integrator.run()
    return success

if __name__ == "__main__":
    main()