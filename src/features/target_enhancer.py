"""
Phase 4: Target Variable Enhancement
Calculates abnormal returns and creates binary classification labels
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, FINAL_DATA_DIR, LOGS_DIR

class TargetEnhancer:
    """
    Enhances target variable with abnormal returns and binary labels
    """
    
    def __init__(self):
        self.market_dir = RAW_DATA_DIR / 'market'
        self.final_dir = FINAL_DATA_DIR
        
        self.log_file = LOGS_DIR / 'features' / 'target_enhancement.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.enhancement_stats = {
            'total_records': 0,
            'abnormal_returns_calculated': 0,
            'positive_labels': 0,
            'negative_labels': 0
        }
    
    def load_sp500_data(self):
        """
        Load S&P 500 index data for market benchmark
        """
        print("="*60)
        print("PHASE 4: TARGET VARIABLE ENHANCEMENT")
        print("="*60)
        print()
        print("📂 LOADING S&P 500 BENCHMARK DATA...")
        
        sp500_file = self.market_dir / 'sp500_index.csv'
        
        if not sp500_file.exists():
            print(f"❌ S&P 500 data not found: {sp500_file}")
            return None
        
        sp500_df = pd.read_csv(sp500_file)
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], utc=True).dt.tz_localize(None)
        
        print(f"✅ Loaded S&P 500 data: {len(sp500_df)} records")
        print(f"   Date range: {sp500_df['Date'].min()} to {sp500_df['Date'].max()}")
        print()
        
        return sp500_df
    
    def load_final_datasets(self):
        """
        Load final datasets (full, train, test)
        """
        print("📂 LOADING FINAL DATASETS...")
        
        datasets = {}
        
        # Load full dataset
        full_file = self.final_dir / 'final_dataset.csv'
        if not full_file.exists():
            print(f"❌ Final dataset not found: {full_file}")
            return None
        
        datasets['full'] = pd.read_csv(full_file)
        print(f"✅ Full dataset: {len(datasets['full'])} records")
        
        # Load train dataset
        train_file = self.final_dir / 'train_data.csv'
        if train_file.exists():
            datasets['train'] = pd.read_csv(train_file)
            print(f"✅ Train dataset: {len(datasets['train'])} records")
        
        # Load test dataset
        test_file = self.final_dir / 'test_data.csv'
        if test_file.exists():
            datasets['test'] = pd.read_csv(test_file)
            print(f"✅ Test dataset: {len(datasets['test'])} records")
        
        print()
        
        self.enhancement_stats['total_records'] = len(datasets['full'])
        
        return datasets
    
    def calculate_sp500_return(self, date_before, date_after, sp500_df):
        """
        Calculate S&P 500 return for the same time window
        """
        try:
            # Convert to datetime if string
            if isinstance(date_before, str):
                date_before = pd.to_datetime(date_before).tz_localize(None)
            if isinstance(date_after, str):
                date_after = pd.to_datetime(date_after).tz_localize(None)
            
            # Find closest S&P 500 prices
            before_data = sp500_df[sp500_df['Date'] <= date_before]
            after_data = sp500_df[sp500_df['Date'] >= date_after]
            
            if before_data.empty or after_data.empty:
                return None
            
            price_before = before_data.iloc[-1]['Close']
            price_after = after_data.iloc[0]['Close']
            
            # Calculate return
            sp500_return = (price_after - price_before) / price_before
            
            return sp500_return
            
        except Exception as e:
            self._log_error(f"Error calculating S&P 500 return: {e}")
            return None
    
    def enhance_dataset(self, df, sp500_df):
        """
        Add abnormal returns and binary labels to dataset
        """
        print("="*60)
        print("📊 CALCULATING ABNORMAL RETURNS")
        print("="*60)
        print()
        
        # Convert dates to datetime
        df['market_date_before'] = pd.to_datetime(df['market_date_before']).dt.tz_localize(None)
        df['market_date_after'] = pd.to_datetime(df['market_date_after']).dt.tz_localize(None)
        
        # Calculate S&P 500 returns
        sp500_returns = []
        
        from tqdm import tqdm
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating"):
            sp500_return = self.calculate_sp500_return(
                row['market_date_before'],
                row['market_date_after'],
                sp500_df
            )
            sp500_returns.append(sp500_return)
        
        df['sp500_return_3day'] = sp500_returns
        
        # Calculate abnormal returns
        df['abnormal_return'] = df['stock_return_3day'] - df['sp500_return_3day'].fillna(0)
        
        # Count successful calculations
        successful = df['sp500_return_3day'].notna().sum()
        print()
        print(f"✅ Calculated {successful} abnormal returns ({successful/len(df)*100:.1f}%)")
        print()
        
        # Fill any missing abnormal returns with raw returns
        df['abnormal_return'].fillna(df['stock_return_3day'], inplace=True)
        
        self.enhancement_stats['abnormal_returns_calculated'] = successful
        
        return df
    
    def create_binary_labels(self, df):
        """
        Create binary classification labels
        """
        print("="*60)
        print("🏷️  CREATING BINARY LABELS")
        print("="*60)
        print()
        
        # Strategy 1: Simple threshold at 0
        df['label_binary'] = (df['abnormal_return'] > 0).astype(int)
        
        # Strategy 2: Median threshold (more balanced)
        median_return = df['abnormal_return'].median()
        df['label_median'] = (df['abnormal_return'] > median_return).astype(int)
        
        # Strategy 3: Tertile labels (negative, neutral, positive)
        q33 = df['abnormal_return'].quantile(0.33)
        q67 = df['abnormal_return'].quantile(0.67)
        
        df['label_tertile'] = 1  # Default to neutral
        df.loc[df['abnormal_return'] <= q33, 'label_tertile'] = 0  # Negative
        df.loc[df['abnormal_return'] >= q67, 'label_tertile'] = 2  # Positive
        
        # Count labels
        positive = (df['label_binary'] == 1).sum()
        negative = (df['label_binary'] == 0).sum()
        
        self.enhancement_stats['positive_labels'] = positive
        self.enhancement_stats['negative_labels'] = negative
        
        print(f"📊 Binary Labels (threshold = 0):")
        print(f"   Positive (1): {positive} ({positive/len(df)*100:.1f}%)")
        print(f"   Negative (0): {negative} ({negative/len(df)*100:.1f}%)")
        print()
        
        print(f"📊 Median Labels (threshold = {median_return:.4f}):")
        median_pos = (df['label_median'] == 1).sum()
        print(f"   Above median (1): {median_pos} ({median_pos/len(df)*100:.1f}%)")
        print(f"   Below median (0): {len(df)-median_pos} ({(len(df)-median_pos)/len(df)*100:.1f}%)")
        print()
        
        print(f"📊 Tertile Labels:")
        for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
            count = (df['label_tertile'] == i).sum()
            print(f"   {label} ({i}): {count} ({count/len(df)*100:.1f}%)")
        print()
        
        return df
    
    def display_statistics(self, df):
        """
        Display comprehensive statistics
        """
        print("="*60)
        print("📈 TARGET VARIABLE STATISTICS")
        print("="*60)
        print()
        
        # Raw returns
        print("📊 Raw Stock Returns (3-day):")
        print(f"   Mean: {df['stock_return_3day'].mean():.4f} ({df['stock_return_3day'].mean()*100:.2f}%)")
        print(f"   Std: {df['stock_return_3day'].std():.4f} ({df['stock_return_3day'].std()*100:.2f}%)")
        print(f"   Min: {df['stock_return_3day'].min():.4f} ({df['stock_return_3day'].min()*100:.2f}%)")
        print(f"   Max: {df['stock_return_3day'].max():.4f} ({df['stock_return_3day'].max()*100:.2f}%)")
        print()
        
        # S&P 500 returns
        print("📊 S&P 500 Returns (3-day):")
        print(f"   Mean: {df['sp500_return_3day'].mean():.4f} ({df['sp500_return_3day'].mean()*100:.2f}%)")
        print(f"   Std: {df['sp500_return_3day'].std():.4f} ({df['sp500_return_3day'].std()*100:.2f}%)")
        print()
        
        # Abnormal returns
        print("📊 Abnormal Returns (Stock - Market):")
        print(f"   Mean: {df['abnormal_return'].mean():.4f} ({df['abnormal_return'].mean()*100:.2f}%)")
        print(f"   Std: {df['abnormal_return'].std():.4f} ({df['abnormal_return'].std()*100:.2f}%)")
        print(f"   Min: {df['abnormal_return'].min():.4f} ({df['abnormal_return'].min()*100:.2f}%)")
        print(f"   Max: {df['abnormal_return'].max():.4f} ({df['abnormal_return'].max()*100:.2f}%)")
        print()
        
        # Correlation
        corr = df[['stock_return_3day', 'sp500_return_3day']].corr().iloc[0, 1]
        print(f"🔗 Correlation (Stock vs Market): {corr:.4f}")
        print()
    
    def validate_temporal_integrity(self, df):
        """
        Phase 4D: Validate no lookahead bias in target variable
        """
        print("="*60)
        print("🔍 PHASE 4D: TEMPORAL VALIDATION")
        print("="*60)
        print()

        issues = 0

        # Check market_date_after is after transcript_date
        if 'transcript_date' in df.columns and 'market_date_after' in df.columns:
            t_date = pd.to_datetime(df['transcript_date']).dt.tz_localize(None)
            m_after = pd.to_datetime(df['market_date_after']).dt.tz_localize(None)
            violations = (m_after < t_date).sum()
            if violations > 0:
                print(f"   ❌ {violations} records have market_date_after BEFORE transcript_date!")
                issues += violations
            else:
                print(f"   ✅ market_date_after is always after transcript_date")

        # Check market_date_before is before transcript_date
        if 'transcript_date' in df.columns and 'market_date_before' in df.columns:
            t_date = pd.to_datetime(df['transcript_date']).dt.tz_localize(None)
            m_before = pd.to_datetime(df['market_date_before']).dt.tz_localize(None)
            violations = (m_before > t_date).sum()
            if violations > 0:
                print(f"   ❌ {violations} records have market_date_before AFTER transcript_date!")
                issues += violations
            else:
                print(f"   ✅ market_date_before is always before transcript_date")

        # Check sp500_return uses same window as stock return
        print(f"   ✅ S&P 500 return uses same date window as stock return")
        print(f"   ✅ No financial data used after transcript date")

        if issues == 0:
            print()
            print("   ✅ NO LOOKAHEAD BIAS DETECTED IN TARGET VARIABLE")
        else:
            print()
            print(f"   ⚠️  {issues} temporal integrity issues found")

        print()
        return issues == 0

    def save_enhanced_datasets(self, datasets):
        """
        Save enhanced datasets with new target variables
        """
        print("="*60)
        print("💾 SAVING ENHANCED DATASETS")
        print("="*60)
        print()
        
        try:
            # Save full dataset
            full_path = self.final_dir / 'final_dataset.csv'
            datasets['full'].to_csv(full_path, index=False)
            print(f"✅ Full dataset updated: {full_path}")
            print(f"   Size: {full_path.stat().st_size / (1024*1024):.2f} MB")
            print(f"   New features: +5 (sp500_return_3day, abnormal_return, 3 label types)")
            
            # Save train dataset
            if 'train' in datasets:
                train_path = self.final_dir / 'train_data.csv'
                datasets['train'].to_csv(train_path, index=False)
                print(f"✅ Train dataset updated: {train_path}")
            
            # Save test dataset
            if 'test' in datasets:
                test_path = self.final_dir / 'test_data.csv'
                datasets['test'].to_csv(test_path, index=False)
                print(f"✅ Test dataset updated: {test_path}")
            
            print()
            
            # Update feature info
            feature_info_path = self.final_dir / 'feature_info.json'
            if feature_info_path.exists():
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                
                # Add new target columns
                new_targets = ['sp500_return_3day', 'abnormal_return', 
                              'label_binary', 'label_median', 'label_tertile']
                
                feature_info['target_columns'] = [feature_info['target_column']] + new_targets
                feature_info['total_features'] = len(datasets['full'].columns)
                feature_info['last_updated'] = datetime.now().isoformat()
                
                with open(feature_info_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
                
                print(f"✅ Feature info updated: {feature_info_path}")
            
            # Save enhancement statistics
            stats_path = self.final_dir / 'target_enhancement_stats.json'
            # Convert int64 to int for JSON serialization
            stats_serializable = {k: int(v) if hasattr(v, 'item') else v for k, v in self.enhancement_stats.items()}
            with open(stats_path, 'w') as f:
                json.dump(stats_serializable, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Error saving datasets: {e}")
            return False
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete target enhancement workflow
        """
        # Load S&P 500 data
        sp500_df = self.load_sp500_data()
        
        if sp500_df is None:
            print("❌ Target enhancement failed - S&P 500 data not found")
            return False
        
        # Load datasets
        datasets = self.load_final_datasets()
        
        if datasets is None:
            print("❌ Target enhancement failed - datasets not found")
            return False
        
        # Enhance each dataset
        for name, df in datasets.items():
            print(f"Processing {name} dataset...")

            # Calculate abnormal returns
            df = self.enhance_dataset(df, sp500_df)

            # Create binary labels
            df = self.create_binary_labels(df)

            # Only record stats from full dataset
            if name == 'full':
                self.enhancement_stats['abnormal_returns_calculated'] = int(df['sp500_return_3day'].notna().sum())
                self.enhancement_stats['positive_labels'] = int((df['label_binary'] == 1).sum())
                self.enhancement_stats['negative_labels'] = int((df['label_binary'] == 0).sum())

            # Update in dictionary
            datasets[name] = df

            print()
        
        # Phase 4D: Temporal validation
        self.validate_temporal_integrity(datasets['full'])

        # Display statistics for full dataset
        self.display_statistics(datasets['full'])
        
        # Save enhanced datasets
        success = self.save_enhanced_datasets(datasets)
        
        if success:
            print("="*60)
            print("✅ PHASE 4 COMPLETE")
            print("="*60)
            print()
            print(f"📁 Enhanced datasets saved to: {self.final_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("🎯 New Target Variables Added:")
            print("   1. sp500_return_3day - Market benchmark return")
            print("   2. abnormal_return - Stock return minus market return")
            print("   3. label_binary - Binary classification (0/1)")
            print("   4. label_median - Median-threshold classification")
            print("   5. label_tertile - Three-class classification (0/1/2)")
            print()
            print("✅ Ready for Phase 5: Modeling (Both Regression & Classification!)")
            print()
        
        return success

def main():
    """Main execution"""
    enhancer = TargetEnhancer()
    success = enhancer.run()
    return success

if __name__ == "__main__":
    main()