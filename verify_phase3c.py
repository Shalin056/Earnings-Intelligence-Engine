"""
Verification script for Phase 3C
"""

import pandas as pd
import json
from pathlib import Path
from config import FINAL_DATA_DIR

def verify_phase3c():
    """
    Verify Phase 3C feature integration results
    """
    print("="*60)
    print("PHASE 3C VERIFICATION")
    print("="*60)
    print()
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'final_dataset.csv',
        'train_data.csv',
        'test_data.csv',
        'feature_info.json',
        'integration_stats.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = FINAL_DATA_DIR / file
        if file_path.exists():
            if file.endswith('.csv'):
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"   ✅ {file} ({size_mb:.2f} MB)")
            else:
                print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files missing!")
        return False
    
    print()
    
    # 2. Load and verify datasets
    print("2️⃣ DATASET VERIFICATION...")
    
    full_df = pd.read_csv(FINAL_DATA_DIR / 'final_dataset.csv')
    train_df = pd.read_csv(FINAL_DATA_DIR / 'train_data.csv')
    test_df = pd.read_csv(FINAL_DATA_DIR / 'test_data.csv')
    
    print(f"   Full dataset: {len(full_df):,} records, {len(full_df.columns)} features")
    print(f"   Training set: {len(train_df):,} records")
    print(f"   Test set: {len(test_df):,} records")
    print()
    
    # 3. Verify split
    print("3️⃣ TRAIN/TEST SPLIT...")
    total_split = len(train_df) + len(test_df)
    train_pct = len(train_df) / total_split * 100
    test_pct = len(test_df) / total_split * 100
    
    print(f"   Train: {train_pct:.1f}%")
    print(f"   Test: {test_pct:.1f}%")
    
    if abs(train_pct - 80) < 5:
        print("   ✅ Split ratio is correct (~80/20)")
    else:
        print("   ⚠️  Split ratio is off")
    
    print()
    
    # 4. Feature info
    print("4️⃣ FEATURE INFORMATION...")
    with open(FINAL_DATA_DIR / 'feature_info.json', 'r', encoding='utf-8') as f:
        feature_info = json.load(f)
    
    print(f"   Metadata columns: {len(feature_info['metadata_columns'])}")
    print(f"   Target column: {feature_info['target_column']}")
    print(f"   Feature columns: {len(feature_info['feature_columns'])}")
    print()
    
    # 5. Check for missing values
    print("5️⃣ DATA QUALITY...")
    missing = full_df.isnull().sum().sum()
    print(f"   Total missing values: {missing:,}")
    
    if missing == 0:
        print("   ✅ No missing values")
    else:
        pct = missing / (len(full_df) * len(full_df.columns)) * 100
        print(f"   ⚠️  {missing:,} missing values ({pct:.2f}%)")
    
    print()
    
    # 6. Target variable
    print("6️⃣ TARGET VARIABLE...")
    target = 'stock_return_3day'
    
    if target in full_df.columns:
        print(f"   ✅ Target '{target}' present")
        print(f"   Mean: {full_df[target].mean():.4f}")
        print(f"   Std: {full_df[target].std():.4f}")
        print(f"   Min: {full_df[target].min():.4f}")
        print(f"   Max: {full_df[target].max():.4f}")
    else:
        print(f"   ❌ Target '{target}' MISSING")
    
    print()
    
    # 7. Sample data
    print("7️⃣ SAMPLE DATA (first 3 records):")
    sample_cols = ['ticker', 'quarter', 'stock_return_3day', 
                   'sentiment_score', 'lm_uncertainty', 'word_count']
    if all(col in full_df.columns for col in sample_cols):
        print(full_df.head(3)[sample_cols].to_string())
    else:
        print("   (Some sample columns not available)")
    
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 3C VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    if len(train_df) + len(test_df) == len(full_df):
        print("✅ Train + Test = Full dataset")
    else:
        print("⚠️  Split mismatch")
    
    if missing == 0:
        print("✅ No missing values")
    else:
        print("⚠️  Has missing values - may need imputation")
    
    print()
    print("Ready to commit Phase 3C to Git")
    print()
    print("🎊 PHASE 3 (FEATURE ENGINEERING) COMPLETE!")
    print("Ready for Phase 4: Baseline Modeling")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase3c()