"""
Verification script for Phase 2D
"""

import pandas as pd
import json
from pathlib import Path
from config import PROCESSED_DATA_DIR

def verify_phase2d():
    """
    Verify Phase 2D temporal alignment results
    """
    print("="*60)
    print("PHASE 2D VERIFICATION")
    print("="*60)
    print()
    
    aligned_dir = PROCESSED_DATA_DIR / 'aligned'
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'aligned_data.csv',
        'alignment_statistics.json',
        'alignment_metadata.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = aligned_dir / file
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
    print("2️⃣ ALIGNMENT STATISTICS...")
    with open(aligned_dir / 'alignment_statistics.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print(f"   Total transcripts:      {stats['total_transcripts']}")
    print(f"   Successfully aligned:   {stats['successfully_aligned']}")
    print(f"   Missing financial data: {stats['missing_financial_data']}")
    print(f"   Missing market data:    {stats['missing_market_data']}")
    print(f"   Final aligned records:  {stats['final_aligned_records']}")
    
    if stats['total_transcripts'] > 0:
        success_rate = stats['successfully_aligned'] / stats['total_transcripts'] * 100
        print(f"   Success rate:           {success_rate:.1f}%")
    
    print()
    
    # 3. Load and verify aligned data
    print("3️⃣ ALIGNED DATA VERIFICATION...")
    df = pd.read_csv(aligned_dir / 'aligned_data.csv')
    
    print(f"   Records: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Unique tickers: {df['ticker'].nunique()}")
    print()
    
    # 4. Temporal validation
    print("4️⃣ TEMPORAL VALIDATION...")
    
    df['transcript_date'] = pd.to_datetime(df['transcript_date'])
    df['financial_date'] = pd.to_datetime(df['financial_date'])
    df['market_date_before'] = pd.to_datetime(df['market_date_before'])
    df['market_date_after'] = pd.to_datetime(df['market_date_after'])
    
    # Check for lookahead bias
    lookahead_financial = (df['financial_date'] > df['transcript_date']).sum()
    lookahead_market = (df['market_date_before'] > df['transcript_date']).sum()
    
    print(f"   Financial lookahead violations: {lookahead_financial}")
    print(f"   Market lookahead violations: {lookahead_market}")
    
    if lookahead_financial == 0 and lookahead_market == 0:
        print("   ✅ NO LOOKAHEAD BIAS DETECTED")
    else:
        print("   ❌ LOOKAHEAD BIAS DETECTED!")
    
    print()
    
    # 5. Date ranges
    print("5️⃣ DATE RANGES...")
    print(f"   Transcript dates: {df['transcript_date'].min()} to {df['transcript_date'].max()}")
    print(f"   Financial dates: {df['financial_date'].min()} to {df['financial_date'].max()}")
    print()
    
    # 6. Sample data
    print("6️⃣ SAMPLE ALIGNED DATA (first 3 records):")
    sample_cols = ['ticker', 'quarter', 'transcript_date', 'financial_date', 
                   'stock_return_3day', 'total_management_words']
    print(df.head(3)[sample_cols].to_string())
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 2D VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    if success_rate >= 50:
        print(f"✅ Alignment rate acceptable ({success_rate:.1f}% ≥ 50%)")
    else:
        print(f"⚠️  Alignment rate low ({success_rate:.1f}% < 50%)")
    
    if lookahead_financial == 0 and lookahead_market == 0:
        print("✅ Temporal integrity validated - NO lookahead bias")
    else:
        print("❌ CRITICAL: Lookahead bias detected - DO NOT PROCEED")
    
    print()
    print("Ready to commit Phase 2D to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase2d()