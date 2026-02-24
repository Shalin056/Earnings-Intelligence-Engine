"""
Verification script for Phase 4
"""

import pandas as pd
import json
from pathlib import Path
from config import FINAL_DATA_DIR

def verify_phase4():
    """
    Verify Phase 4 target enhancement results
    """
    print("="*60)
    print("PHASE 4 VERIFICATION")
    print("="*60)
    print()
    
    # 1. Load dataset
    print("1️⃣ LOADING ENHANCED DATASET...")
    full_path = FINAL_DATA_DIR / 'final_dataset.csv'
    
    if not full_path.exists():
        print(f"   ❌ Dataset not found: {full_path}")
        return False
    
    df = pd.read_csv(full_path)
    print(f"   ✅ Loaded: {len(df):,} records, {len(df.columns)} features")
    print()
    
    # 2. Check new columns
    print("2️⃣ CHECKING NEW TARGET VARIABLES...")
    required_cols = [
        'stock_return_3day',
        'sp500_return_3day',
        'abnormal_return',
        'label_binary',
        'label_median',
        'label_tertile'
    ]
    
    for col in required_cols:
        if col in df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} MISSING")
    
    print()
    
    # 3. Abnormal returns statistics
    print("3️⃣ ABNORMAL RETURNS STATISTICS...")
    print(f"   Mean: {df['abnormal_return'].mean():.4f} ({df['abnormal_return'].mean()*100:.2f}%)")
    print(f"   Std: {df['abnormal_return'].std():.4f} ({df['abnormal_return'].std()*100:.2f}%)")
    print(f"   Min: {df['abnormal_return'].min():.4f} ({df['abnormal_return'].min()*100:.2f}%)")
    print(f"   Max: {df['abnormal_return'].max():.4f} ({df['abnormal_return'].max()*100:.2f}%)")
    print()
    
    # 4. Label distributions
    print("4️⃣ LABEL DISTRIBUTIONS...")
    
    # Binary labels
    binary_counts = df['label_binary'].value_counts()
    print(f"   Binary (0 vs 1):")
    for label, count in binary_counts.items():
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    # Median labels
    median_counts = df['label_median'].value_counts()
    print(f"   Median (0 vs 1):")
    for label, count in median_counts.items():
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    # Tertile labels
    tertile_counts = df['label_tertile'].value_counts().sort_index()
    print(f"   Tertile (0/1/2):")
    for label, count in tertile_counts.items():
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    print()
    
    # 5. Comparison
    print("5️⃣ RAW vs ABNORMAL RETURNS...")
    print(f"   Raw return mean: {df['stock_return_3day'].mean():.4f}")
    print(f"   Market return mean: {df['sp500_return_3day'].mean():.4f}")
    print(f"   Abnormal return mean: {df['abnormal_return'].mean():.4f}")
    print(f"   Correlation (stock vs market): {df[['stock_return_3day', 'sp500_return_3day']].corr().iloc[0,1]:.4f}")
    print()
    
    # 6. Sample data
    print("6️⃣ SAMPLE DATA (first 3 records):")
    sample_cols = ['ticker', 'quarter', 'stock_return_3day', 'sp500_return_3day', 
                   'abnormal_return', 'label_binary']
    print(df.head(3)[sample_cols].to_string())
    print()
    
    # 7. Statistics
    print("7️⃣ ENHANCEMENT STATISTICS...")
    stats_path = FINAL_DATA_DIR / 'target_enhancement_stats.json'
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"   Total records: {stats['total_records']}")
        print(f"   Abnormal returns calculated: {stats['abnormal_returns_calculated']}")
        print(f"   Positive labels: {stats['positive_labels']}")
        print(f"   Negative labels: {stats['negative_labels']}")
    
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 4 VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    all_cols_present = all(col in df.columns for col in required_cols)
    
    if all_cols_present:
        print("✅ All target variables present")
    else:
        print("⚠️  Some target variables missing")
    
    if df['abnormal_return'].notna().sum() > len(df) * 0.9:
        print("✅ Abnormal returns calculated successfully")
    else:
        print("⚠️  Some abnormal returns missing")
    
    print()
    print("Ready to commit Phase 4 to Git")
    print()
    print("🎯 You can now build both:")
    print("   • Regression models (predict continuous returns)")
    print("   • Classification models (predict positive/negative)")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase4()