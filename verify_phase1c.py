"""
Verification script for Phase 1C data
"""

import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR

def verify_financial_data():
    """
    Verify the collected financial data
    """
    print("="*60)
    print("PHASE 1C DATA VERIFICATION")
    print("="*60)
    print()
    
    financial_dir = RAW_DATA_DIR / 'financial'
    
    # Check if files exist
    files_to_check = [
        'income_statements.csv',
        'balance_sheets.csv',
        'cashflows.csv',
        'financial_features.csv',
        'collection_log.csv'
    ]
    
    print("1️⃣ CHECKING FILES...")
    all_exist = True
    for file in files_to_check:
        file_path = financial_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   ✅ {file} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {file} MISSING")
            all_exist = False
    print()
    
    if not all_exist:
        print("❌ Some files are missing!")
        return
    
    # Load and verify financial features
    print("2️⃣ VERIFYING FINANCIAL FEATURES...")
    features_df = pd.read_csv(financial_dir / 'financial_features.csv')
    
    print(f"   ✅ Records loaded: {len(features_df):,}")
    print(f"   ✅ Unique tickers: {features_df['Ticker'].nunique()}")
    print(f"   ✅ Total features: {len(features_df.columns) - 2}")
    print()
    
    # Date range
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    print(f"   ✅ Date range: {features_df['Date'].min()} to {features_df['Date'].max()}")
    print()
    
    # Feature completeness
    print("3️⃣ FEATURE COMPLETENESS:")
    completeness = (features_df.notna().sum() / len(features_df) * 100).round(1)
    
    # Group by category
    income_features = [col for col in features_df.columns if any(x in col for x in ['Revenue', 'Profit', 'Income', 'EBIT', 'EPS', 'Expense', 'Tax'])]
    balance_features = [col for col in features_df.columns if any(x in col for x in ['Assets', 'Liabilities', 'Equity', 'Cash', 'Debt', 'Working', 'Inventories'])]
    cashflow_features = [col for col in features_df.columns if any(x in col for x in ['Cash_Flow', 'Capital', 'Dividend', 'Repurchase', 'Change'])]
    
    print("\n   Income Statement Features:")
    for col in income_features:
        print(f"      {col}: {completeness[col]:.1f}%")
    
    print("\n   Balance Sheet Features:")
    for col in balance_features:
        print(f"      {col}: {completeness[col]:.1f}%")
    
    print("\n   Cash Flow Features:")
    for col in cashflow_features:
        print(f"      {col}: {completeness[col]:.1f}%")
    
    print()
    
    # Overall completeness
    avg_completeness = completeness.mean()
    print(f"   📊 Average completeness: {avg_completeness:.1f}%")
    print()
    
    # Sample data
    print("4️⃣ SAMPLE DATA (first 3 records):")
    print(features_df.head(3)[['Ticker', 'Date', 'Total_Revenue', 'Net_Income', 'Total_Assets', 'Operating_Cash_Flow']].to_string())
    print()
    
    # Collection log
    print("5️⃣ COLLECTION LOG:")
    log_df = pd.read_csv(financial_dir / 'collection_log.csv')
    successful = log_df['Successful_Tickers'].dropna()
    failed = log_df['Failed_Tickers'].dropna()
    
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    print(f"   📊 Success Rate: {len(successful)/(len(successful)+len(failed))*100:.1f}%")
    
    if len(failed) > 0:
        print(f"\n   Failed tickers: {', '.join(failed.head(10).tolist())}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    print()
    
    print("="*60)
    print("✅ PHASE 1C VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    # Check thresholds
    success_rate = len(successful)/(len(successful)+len(failed))*100
    
    if success_rate >= 85:
        print("✅ Success rate meets ≥85% threshold")
    else:
        print(f"⚠️  Success rate ({success_rate:.1f}%) below 85% threshold")
    
    if avg_completeness >= 70:
        print("✅ Feature completeness is acceptable (≥70%)")
    else:
        print(f"⚠️  Feature completeness ({avg_completeness:.1f}%) may be low")
    
    print()
    print("Ready to proceed to Phase 1D: Earnings Call Transcripts")
    print()

if __name__ == "__main__":
    verify_financial_data()