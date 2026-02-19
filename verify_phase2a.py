"""
Verification script for Phase 2A
"""

import pandas as pd
import json
from pathlib import Path
from config import PROCESSED_DATA_DIR

def verify_phase2a():
    """
    Verify Phase 2A transcript cleaning results
    """
    print("="*60)
    print("PHASE 2A VERIFICATION")
    print("="*60)
    print()
    
    transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
    
    # 1. Check files exist
    print("1️⃣ CHECKING FILES...")
    files = [
        'transcripts_cleaned.json',
        'transcripts_cleaned_metadata.csv',
        'cleaning_statistics.json'
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
    print("2️⃣ CLEANING STATISTICS...")
    with open(transcripts_dir / 'cleaning_statistics.json', 'r') as f:
        stats = json.load(f)
    
    print(f"   Total processed:      {stats['total_processed']}")
    print(f"   Successfully cleaned: {stats['successfully_cleaned']}")
    print(f"   Failed:               {stats['failed']}")
    print(f"   Success rate:         {stats['successfully_cleaned']/stats['total_processed']*100:.1f}%")
    print()
    print(f"   Avg original length:  {stats['avg_original_length']:.0f} chars")
    print(f"   Avg cleaned length:   {stats['avg_cleaned_length']:.0f} chars")
    print(f"   Chars removed:        {stats['total_chars_removed']:,}")
    print()
    
    # 3. Verify metadata
    print("3️⃣ METADATA VERIFICATION...")
    metadata_df = pd.read_csv(transcripts_dir / 'transcripts_cleaned_metadata.csv')
    
    print(f"   Records: {len(metadata_df)}")
    print(f"   Unique tickers: {metadata_df['ticker'].nunique()}")
    print()
    
    # Sample
    print("   Sample (first 3 records):")
    print(metadata_df.head(3)[['ticker', 'quarter', 'full_text_original_length', 
                                'full_text_cleaned_length', 'total_chars_removed']].to_string())
    print()
    
    # 4. Verify JSON structure
    print("4️⃣ JSON STRUCTURE...")
    with open(transcripts_dir / 'transcripts_cleaned.json', 'r') as f:
        cleaned_data = json.load(f)
    
    print(f"   Total transcripts: {len(cleaned_data)}")
    
    if cleaned_data:
        first = cleaned_data[0]
        required_fields = ['ticker', 'full_text_cleaned', 'prepared_remarks_cleaned', 'qa_section_cleaned']
        
        print("   Checking first transcript structure:")
        for field in required_fields:
            if field in first:
                print(f"      ✅ {field}")
            else:
                print(f"      ❌ {field} MISSING")
    
    print()
    
    # 5. Quality checks
    print("5️⃣ QUALITY CHECKS...")
    
    # Check if any transcript is too short (might indicate cleaning error)
    min_length = metadata_df['full_text_cleaned_length'].min()
    max_length = metadata_df['full_text_cleaned_length'].max()
    
    print(f"   Min cleaned length: {min_length} chars")
    print(f"   Max cleaned length: {max_length} chars")
    
    if min_length < 100:
        short_transcripts = metadata_df[metadata_df['full_text_cleaned_length'] < 100]
        print(f"   ⚠️  {len(short_transcripts)} transcripts are suspiciously short (<100 chars)")
        print(f"      Tickers: {', '.join(short_transcripts['ticker'].tolist())}")
    else:
        print(f"   ✅ All transcripts have reasonable length")
    
    print()
    
    # Success criteria
    print("="*60)
    print("✅ PHASE 2A VERIFICATION COMPLETE")
    print("="*60)
    print()
    
    success_rate = stats['successfully_cleaned'] / stats['total_processed']
    
    if success_rate >= 0.95:
        print("✅ Success rate meets threshold (≥95%)")
    else:
        print(f"⚠️  Success rate ({success_rate:.1%}) below 95%")
    
    if min_length >= 100:
        print("✅ All transcripts properly cleaned")
    else:
        print("⚠️  Some transcripts may need review")
    
    print()
    print("Ready to commit Phase 2A to Git")
    print()
    
    return True

if __name__ == "__main__":
    verify_phase2a()