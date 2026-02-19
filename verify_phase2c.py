"""
Verification script for Phase 2C
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from config import PROCESSED_DATA_DIR

def verify_phase2c():
    """
    Verify Phase 2C financial normalization results
    """
    print("="*60)
    print("PHASE 2C VERIFICATION")
    print("="*60)
    print()
    
    financial_dir = PROCESSED_DATA_DIR / 'financial'
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'financial_features_normalized.csv',
        'robust_scaler.pkl',
        'normalization_metadata.json',
        'normalization_statistics.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = financial_dir / file
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
    
    # 2. Load and verify statistics
    print("2️⃣ NORMALIZATION STATISTICS...")
    with open(financial_dir / 'normalization_statistics.json', 'r') as f:
        stats = json.load(f)
    
    print(f"   Total records:           {stats['total_records']:,}")
    print(f"   Successfully normalized: {stats['successfully_normalized']:,}")
    print(f"   Features normalized:     {stats['features_normalized']}")
    print(f"   Records with missing:    {stats['records_with_missing_values']}")
    print(f"   Outliers detected:       {stats['outliers_detected']:,}")
    print()
    
    # 3. Verify normalized data
    print("3️⃣ NORMALIZED DATA VERIFICATION...")
    df = pd.read_csv(financial_dir / 'financial_features_normalized.csv')
    
    print(f"   Records: {len(df):,}")
    print(f"   Features: {len(df.columns) - 2}")  # Exclude Ticker, Date
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    
    # 4. Check normalization quality
    print("4️⃣ NORMALIZATION QUALITY...")
    
    # Get financial features (exclude metadata)
    feature_cols = [col for col in df.columns if col not in ['Ticker', 'Date']]
    
    # Check if features are normalized (mean ≈ 0, std ≈ 1 for RobustScaler)
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    print(f"   Mean of means: {means.mean():.3f} (should be close to 0)")
    print(f"   Mean of stds: {stds.mean():.3f}")
    print()
    
    # Check for missing values
    missing = df[feature_cols].isnull().sum().sum()
    print(f"   Missing values: {missing} (should be 0)")
    
    if missing == 0:
        print("   ✅ No missing values in normalized data")
    else:
        print(f"   ⚠️  {missing} missing values found")
    
    print()
    
    # 5. Verify scaler
    print("5️⃣ SCALER VERIFICATION...")
    scaler = joblib.load(financial_dir / 'robust_scaler.pkl')
    print(f"   Scaler type: {type(scaler).__name__}")
    print(f"   Features fitted: {scaler.n_features_in_}")
    print("   ✅ Scaler loaded successfully")
    print()
    
    # 6. Sample data
    print("6️⃣ SAMPLE NORMALIZED DATA (first 3 records):")
    sample_cols = ['Ticker', 'Date'] + feature_cols[:3]
    print(df.head(3)[sample_cols].to_string())
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 2C VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    if stats['successfully_normalized'] == stats['total_records']:
        print("✅ All records successfully normalized")
    else:
        print(f"⚠️  {stats['total_records'] - stats['successfully_normalized']} records failed")
    
    if missing == 0:
        print("✅ No missing values in normalized data")
    else:
        print("⚠️  Missing values need attention")
    
    print()
    print("Ready to commit Phase 2C to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase2c()