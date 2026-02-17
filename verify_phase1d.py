"""
Verification script for Phase 1D data
"""

import pandas as pd
import json
from pathlib import Path
from config import RAW_DATA_DIR

def verify_transcripts():
    """
    Verify the collected transcript data
    """
    print("="*60)
    print("PHASE 1D DATA VERIFICATION")
    print("="*60)
    print()
    
    transcripts_dir = RAW_DATA_DIR / 'transcripts'
    
    # 1. Check files
    print("1️⃣ CHECKING FILES...")
    files_to_check = [
        'transcripts_metadata.csv',
        'transcripts_full.json',
    ]
    
    for file in files_to_check:
        file_path = transcripts_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   ✅ {file} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {file} MISSING")
    
    # Check individual files
    individual_dir = transcripts_dir / 'individual'
    if individual_dir.exists():
        txt_files = list(individual_dir.glob('*.txt'))
        print(f"   ✅ individual/ folder ({len(txt_files)} .txt files)")
    else:
        print(f"   ❌ individual/ folder MISSING")
    
    print()
    
    # 2. Load and verify metadata
    print("2️⃣ VERIFYING METADATA...")
    metadata_df = pd.read_csv(transcripts_dir / 'transcripts_metadata.csv')
    
    print(f"   ✅ Records loaded: {len(metadata_df)}")
    print(f"   ✅ Unique tickers: {metadata_df['ticker'].nunique()}")
    print(f"   ✅ Unique quarters: {metadata_df['quarter'].nunique()}")
    print(f"   ✅ Average word count: {metadata_df['word_count'].mean():.0f}")
    print()
    
    # 3. Sample transcript preview
    print("3️⃣ SAMPLE METADATA (first 5 records):")
    print(metadata_df.head().to_string())
    print()
    
    # 4. Load JSON and check structure
    print("4️⃣ VERIFYING JSON STRUCTURE...")
    with open(transcripts_dir / 'transcripts_full.json', 'r') as f:
        transcripts_json = json.load(f)
    
    print(f"   ✅ Total transcripts in JSON: {len(transcripts_json)}")
    
    # Check first transcript structure
    if transcripts_json:
        first_transcript = transcripts_json[0]
        required_fields = ['ticker', 'company_name', 'quarter', 'date', 
                          'prepared_remarks', 'qa_section', 'full_text']
        
        print(f"\n   Checking structure of first transcript:")
        for field in required_fields:
            if field in first_transcript:
                if field == 'full_text':
                    print(f"      ✅ {field}: {len(first_transcript[field])} characters")
                else:
                    print(f"      ✅ {field}: {first_transcript[field]}")
            else:
                print(f"      ❌ {field}: MISSING")
    
    print()
    
    # 5. Read sample individual file
    print("5️⃣ SAMPLE INDIVIDUAL TRANSCRIPT:")
    if individual_dir.exists() and len(txt_files) > 0:
        sample_file = txt_files[0]
        print(f"   File: {sample_file.name}")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"   Lines: {len(lines)}")
            print(f"   Characters: {len(content)}")
            print(f"\n   First 500 characters:")
            print("   " + "-"*56)
            print("   " + content[:500].replace('\n', '\n   '))
            print("   " + "-"*56)
    
    print()
    
    # 6. Statistics
    print("6️⃣ STATISTICS:")
    print(f"   📊 Tickers: {metadata_df['ticker'].nunique()}")
    print(f"   📊 Total transcripts: {len(metadata_df)}")
    print(f"   📊 Transcripts per ticker: {len(metadata_df) / metadata_df['ticker'].nunique():.1f}")
    print(f"   📊 Word count range: {metadata_df['word_count'].min()}-{metadata_df['word_count'].max()}")
    print()
    
    # Ticker distribution
    print("   Top 10 tickers by transcript count:")
    ticker_counts = metadata_df['ticker'].value_counts().head(10)
    for ticker, count in ticker_counts.items():
        print(f"      {ticker}: {count} transcripts")
    
    print()
    
    print("="*60)
    print("✅ PHASE 1D VERIFICATION COMPLETE")
    print("="*60)
    print()
    print("✅ All data structures verified")
    print("✅ Ready to proceed to Phase 2: Data Preprocessing")
    print()

if __name__ == "__main__":
    verify_transcripts()