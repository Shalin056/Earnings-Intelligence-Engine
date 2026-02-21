"""
Verification script for Phase 3A
"""

import pandas as pd
import json
from pathlib import Path
from config import FEATURES_DIR

def verify_phase3a():
    """
    Verify Phase 3A FinBERT feature extraction results
    """
    print("="*60)
    print("PHASE 3A VERIFICATION")
    print("="*60)
    print()
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'finbert_features.csv',
        'finbert_features_metadata.csv',
        'finbert_extraction_stats.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = FEATURES_DIR / file
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
    
    # 2. Load and verify features
    print("2️⃣ FEATURE VERIFICATION...")
    features_df = pd.read_csv(FEATURES_DIR / 'finbert_features.csv')
    
    print(f"   Records: {len(features_df):,}")
    print(f"   Total features: {len(features_df.columns)}")
    print()
    
    # Check for embeddings (should be 768 columns named embedding_000 to embedding_767)
    embedding_cols = [col for col in features_df.columns if col.startswith('embedding_')]
    print(f"   Embedding dimensions: {len(embedding_cols)}")
    
    if len(embedding_cols) == 768:
        print("   ✅ All 768 embedding dimensions present")
    else:
        print(f"   ⚠️  Expected 768 embeddings, found {len(embedding_cols)}")
    
    print()
    
    # 3. Sentiment features
    print("3️⃣ SENTIMENT FEATURES...")
    sentiment_cols = [
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'sentiment_score',
        'mgmt_sentiment_score', 'analyst_sentiment_score', 'sentiment_divergence'
    ]
    
    for col in sentiment_cols:
        if col in features_df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} MISSING")
    
    print()
    
    # 4. Statistics
    print("4️⃣ SENTIMENT STATISTICS...")
    print(f"   Average sentiment score: {features_df['sentiment_score'].mean():.3f}")
    print(f"   Sentiment range: [{features_df['sentiment_score'].min():.3f}, {features_df['sentiment_score'].max():.3f}]")
    print(f"   Positive transcripts: {(features_df['sentiment_score'] > 0).sum()} ({(features_df['sentiment_score'] > 0).sum()/len(features_df)*100:.1f}%)")
    print(f"   Negative transcripts: {(features_df['sentiment_score'] < 0).sum()} ({(features_df['sentiment_score'] < 0).sum()/len(features_df)*100:.1f}%)")
    print()
    
    # 5. Sample data
    print("5️⃣ SAMPLE FEATURES (first 3 records):")
    sample_cols = ['ticker', 'quarter', 'sentiment_score', 'mgmt_sentiment_score', 
                   'analyst_sentiment_score', 'sentiment_divergence']
    print(features_df.head(3)[sample_cols].to_string())
    print()
    
    # 6. Extraction stats
    print("6️⃣ EXTRACTION STATISTICS...")
    with open(FEATURES_DIR / 'finbert_extraction_stats.json', 'r') as f:
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
    print("✅ PHASE 3A VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    if len(embedding_cols) == 768:
        print("✅ FinBERT embeddings extracted successfully")
    else:
        print("⚠️  Embedding dimensions incomplete")
    
    if stats['successfully_extracted'] == stats['total_transcripts']:
        print("✅ All transcripts processed successfully")
    else:
        print(f"⚠️  {stats['failed']} transcripts failed")
    
    print()
    print("Ready to commit Phase 3A to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase3a()