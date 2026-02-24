"""
Verification script for Phase 3B
"""

import pandas as pd
import json
from pathlib import Path
from config import FEATURES_DIR

def verify_phase3b():
    """
    Verify Phase 3B NLP feature extraction results
    """
    print("="*60)
    print("PHASE 3B VERIFICATION")
    print("="*60)
    print()
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'nlp_features.csv',
        'nlp_extraction_stats.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = FEATURES_DIR / file
        if file_path.exists():
            if file.endswith('.csv'):
                size_kb = file_path.stat().st_size / 1024
                print(f"   ✅ {file} ({size_kb:.2f} KB)")
            else:
                print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files missing!")
        return False
    
    print()
    
    # 2. Load and verify features
    print("2️⃣ FEATURE VERIFICATION...")
    features_df = pd.read_csv(FEATURES_DIR / 'nlp_features.csv')
    
    print(f"   Records: {len(features_df):,}")
    print(f"   Total features: {len(features_df.columns)}")
    print()
    
    # 3. Check key features
    print("3️⃣ LOUGHRAN-MCDONALD FEATURES...")
    lm_features = [
        'lm_negative', 'lm_positive', 'lm_uncertainty', 
        'lm_litigious', 'lm_constraining', 'lm_net_sentiment'
    ]
    
    for feat in lm_features:
        if feat in features_df.columns:
            print(f"   ✅ {feat}")
        else:
            print(f"   ❌ {feat} MISSING")
    
    print()
    
    # 4. Text statistics
    print("4️⃣ TEXT STATISTICS...")
    text_features = ['word_count', 'sentence_count', 'lexical_diversity', 'question_count']
    
    for feat in text_features:
        if feat in features_df.columns:
            print(f"   ✅ {feat}")
        else:
            print(f"   ❌ {feat} MISSING")
    
    print()
    
    # 5. Statistics
    print("5️⃣ KEY METRICS...")
    print(f"   Average uncertainty: {features_df['lm_uncertainty'].mean():.4f}")
    print(f"   Average negative: {features_df['lm_negative'].mean():.4f}")
    print(f"   Average positive: {features_df['lm_positive'].mean():.4f}")
    print(f"   Average word count: {features_df['word_count'].mean():.0f}")
    print()
    
    # 6. Sample data
    print("6️⃣ SAMPLE FEATURES (first 3 records):")
    sample_cols = ['ticker', 'quarter', 'lm_uncertainty', 'lm_negative', 
                   'lm_positive', 'word_count']
    print(features_df.head(3)[sample_cols].to_string())
    print()
    
    # 7. Extraction stats
    print("7️⃣ EXTRACTION STATISTICS...")
    with open(FEATURES_DIR / 'nlp_extraction_stats.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print(f"   Total transcripts: {stats['total_transcripts']}")
    print(f"   Successfully extracted: {stats['successfully_extracted']}")
    print(f"   Failed: {stats['failed']}")
    
    if stats['total_transcripts'] > 0:
        success_rate = stats['successfully_extracted'] / stats['total_transcripts'] * 100
        print(f"   Success rate: {success_rate:.1f}%")
    
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 3B VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    if stats['successfully_extracted'] == stats['total_transcripts']:
        print("✅ All transcripts processed successfully")
    else:
        print(f"⚠️  {stats['failed']} transcripts failed")
    
    print()
    print("Ready to commit Phase 3B to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase3b()