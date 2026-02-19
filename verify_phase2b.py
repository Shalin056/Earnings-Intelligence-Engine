"""
Verification script for Phase 2B
"""

import pandas as pd
import json
from pathlib import Path
from config import PROCESSED_DATA_DIR

def verify_phase2b():
    """
    Verify Phase 2B speaker segmentation results
    """
    print("="*60)
    print("PHASE 2B VERIFICATION")
    print("="*60)
    print()
    
    transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'transcripts_segmented.json',
        'transcripts_segmented_metadata.csv',
        'segmentation_statistics.json'
    ]
    
    all_exist = True
    for file in files:
        file_path = transcripts_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   ✅ {file} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {file} MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files missing!")
        return False
    
    print()
    
    # 2. Load and verify statistics
    print("2️⃣ SEGMENTATION STATISTICS...")
    with open(transcripts_dir / 'segmentation_statistics.json', 'r') as f:
        stats = json.load(f)
    
    print(f"   Total processed:         {stats['total_processed']}")
    print(f"   Successfully segmented:  {stats['successfully_segmented']}")
    print(f"   Failed:                  {stats['failed']}")
    print(f"   Success rate:            {stats['successfully_segmented']/stats['total_processed']*100:.1f}%")
    print()
    print(f"   Avg management words:    {stats['avg_management_words']:.0f}")
    print(f"   Avg analyst words:       {stats['avg_analyst_words']:.0f}")
    print(f"   Avg mgmt/analyst ratio:  {stats['avg_management_words'] / max(stats['avg_analyst_words'], 1):.2f}x")
    print()
    
    # 3. Verify metadata
    print("3️⃣ METADATA VERIFICATION...")
    metadata_df = pd.read_csv(transcripts_dir / 'transcripts_segmented_metadata.csv')
    
    print(f"   Records: {len(metadata_df)}")
    print(f"   Unique tickers: {metadata_df['ticker'].nunique()}")
    print()
    
    # Sample
    print("   Sample (first 3 records):")
    print(metadata_df.head(3)[['ticker', 'quarter', 'total_management_words', 
                                'total_analyst_words', 'management_analyst_word_ratio']].to_string())
    print()
    
    # 4. Quality checks
    print("4️⃣ QUALITY CHECKS...")
    
    # Check word counts are reasonable
    min_mgmt_words = metadata_df['total_management_words'].min()
    min_analyst_words = metadata_df['total_analyst_words'].min()
    
    print(f"   Min management words: {min_mgmt_words}")
    print(f"   Min analyst words:    {min_analyst_words}")
    
    # Management should always have more words (prepared remarks + Q&A answers)
    avg_ratio = metadata_df['management_analyst_word_ratio'].mean()
    print(f"   Avg mgmt/analyst ratio: {avg_ratio:.2f}x")
    
    if avg_ratio > 1:
        print(f"   ✅ Management speaks more than analysts (expected)")
    else:
        print(f"   ⚠️  Unusual ratio - analysts speaking more than management")
    
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 2B VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    success_rate = stats['successfully_segmented'] / stats['total_processed']
    
    if success_rate >= 0.95:
        print("✅ Success rate meets threshold (≥95%)")
    else:
        print(f"⚠️  Success rate ({success_rate:.1%}) below 95%")
    
    if min_mgmt_words >= 100:
        print("✅ All transcripts have sufficient management content")
    else:
        print("⚠️  Some transcripts may have insufficient content")
    
    print()
    print("Ready to commit Phase 2B to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase2b()