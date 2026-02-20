# """
# Phase 1D Fixed - Generate transcripts matching actual financial data
# """

# import pandas as pd
# import json
# from pathlib import Path
# from datetime import datetime
# from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# def generate_aligned_transcripts():
#     """
#     Generate sample transcripts that match actual financial data
#     """
#     print("="*60)
#     print("PHASE 1D: GENERATING ALIGNED TRANSCRIPTS")
#     print("="*60)
#     print()
    
#     # 1. Load financial data to get available tickers and dates
#     financial_file = PROCESSED_DATA_DIR / 'financial' / 'financial_features_normalized.csv'
    
#     if not financial_file.exists():
#         print(f"❌ Financial data not found: {financial_file}")
#         print("   Please run Phase 1C first")
#         return False
    
#     print(f"📂 Reading financial data...")
#     financials_df = pd.read_csv(financial_file)
    
#     print(f"✅ Loaded {len(financials_df)} financial records")
#     print(f"   Tickers: {financials_df['Ticker'].nunique()}")
#     print()
    
#     # 2. Get top 50 tickers with most data
#     ticker_counts = financials_df['Ticker'].value_counts()
#     top_50_tickers = ticker_counts.head(50).index.tolist()
    
#     print(f"📊 Using top 50 tickers:")
#     print(f"   {', '.join(top_50_tickers[:10])}...")
#     print()
    
#     # 3. Generate 3 transcripts per ticker (150 total)
#     quarters = ['Q2 2024', 'Q3 2024', 'Q4 2024']
#     dates = ['2024-07-31', '2024-10-31', '2025-01-31']
    
#     transcripts = []
    
#     for ticker in top_50_tickers:
#         for i, (quarter, date) in enumerate(zip(quarters, dates)):
#             transcript = {
#                 'ticker': ticker,
#                 'company_name': f"{ticker} Inc.",
#                 'quarter': quarter,
#                 'date': date,
#                 'fiscal_year': 2024,
#                 'fiscal_quarter': i + 2,  # Q2, Q3, Q4
                
#                 'prepared_remarks': f"Good afternoon, and thank you for joining {ticker}'s {quarter} earnings call. We delivered strong results this quarter with revenue growth across key segments. Our operational efficiency continues to improve, and we're well-positioned for the remainder of the year.",
                
#                 'qa_section': f"Analyst: Can you provide more details on the revenue growth?\n\nManagement: Thanks for the question. The revenue growth was driven by multiple factors including strong customer demand and operational improvements. We're seeing positive trends across all business segments.",
                
#                 'full_text': '',
#                 'word_count': 0,
#                 'speaker_count': 2,
#                 'source': 'SAMPLE_ALIGNED',
#                 'collection_date': datetime.now().strftime('%Y-%m-%d')
#             }
            
#             # Combine full text
#             transcript['full_text'] = transcript['prepared_remarks'] + "\n\n" + transcript['qa_section']
#             transcript['word_count'] = len(transcript['full_text'].split())
            
#             transcripts.append(transcript)
    
#     print(f"✅ Generated {len(transcripts)} transcripts")
#     print(f"   Tickers: {len(set(t['ticker'] for t in transcripts))}")
#     print(f"   Quarters: {set(t['quarter'] for t in transcripts)}")
#     print()
    
#     # 4. Save transcripts
#     output_dir = RAW_DATA_DIR / 'transcripts'
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Save metadata
#     metadata_df = pd.DataFrame([{
#         'ticker': t['ticker'],
#         'company_name': t['company_name'],
#         'quarter': t['quarter'],
#         'date': t['date'],
#         'fiscal_year': t['fiscal_year'],
#         'fiscal_quarter': t['fiscal_quarter'],
#         'word_count': t['word_count'],
#         'source': t['source'],
#         'collection_date': t['collection_date']
#     } for t in transcripts])
    
#     metadata_path = output_dir / 'transcripts_metadata.csv'
#     metadata_df.to_csv(metadata_path, index=False)
#     print(f"✅ Metadata saved: {metadata_path}")
    
#     # Save full JSON
#     json_path = output_dir / 'transcripts_full.json'
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(transcripts, f, indent=2, ensure_ascii=False)
#     print(f"✅ Full transcripts saved: {json_path}")
    
#     # Save individual files
#     individual_dir = output_dir / 'individual'
#     individual_dir.mkdir(exist_ok=True)
    
#     for transcript in transcripts:
#         filename = f"{transcript['ticker']}_{transcript['quarter'].replace(' ', '_')}.txt"
#         file_path = individual_dir / filename
        
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(f"Company: {transcript['company_name']}\n")
#             f.write(f"Ticker: {transcript['ticker']}\n")
#             f.write(f"Quarter: {transcript['quarter']}\n")
#             f.write(f"Date: {transcript['date']}\n")
#             f.write("="*60 + "\n\n")
#             f.write(transcript['full_text'])
    
#     print(f"✅ Individual files saved: {individual_dir}/")
#     print()
    
#     print("="*60)
#     print("✅ PHASE 1D COMPLETE")
#     print("="*60)
#     print()
#     print("📊 Summary:")
#     print(f"   Total transcripts: {len(transcripts)}")
#     print(f"   Unique tickers: {len(set(t['ticker'] for t in transcripts))}")
#     print(f"   Date range: {min(t['date'] for t in transcripts)} to {max(t['date'] for t in transcripts)}")
#     print()
#     print("✅ Transcripts are now aligned with financial data!")
#     print()
    
#     return True

# if __name__ == "__main__":
#     success = generate_aligned_transcripts()
    
#     if success:
#         print("Next steps:")
#         print("1. Re-run Phase 2A: python run_phase2a.py")
#         print("2. Re-run Phase 2B: python run_phase2b.py")  
#         print("3. Run Phase 2D: python run_phase2d.py")
#         print("\nPhase 2D should now succeed!")

"""
Phase 1D Fixed - Generate transcripts matching actual financial data
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def update_market_data():
    """Update market data to cover 2024-2025 transcript date range."""
    import yfinance as yf
    from tqdm import tqdm

    market_file = RAW_DATA_DIR / 'market' / 'stock_prices.csv'
    if not market_file.exists():
        print("⚠️  Market data file not found, skipping update")
        return

    print("="*60)
    print("📡 UPDATING MARKET DATA TO 2024-2025")
    print("="*60)

    existing_df = pd.read_csv(market_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True).dt.tz_convert(None)
    current_end = existing_df['Date'].max()
    print(f"   Current data ends: {current_end.date()}")

    if current_end >= pd.Timestamp('2025-01-01'):
        print("   ✅ Market data already up to date, skipping")
        return

    tickers = existing_df['Ticker'].unique().tolist()
    START_DATE = (current_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    END_DATE = "2026-02-20"
    print(f"   Downloading: {START_DATE} to {END_DATE} for {len(tickers)} tickers")

    new_records = []
    failed = []
    for ticker in tqdm(tickers, desc="Updating market data"):
        try:
            df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=False)
            if df.empty:
                failed.append(ticker)
                continue
            df = df.reset_index()
            df['Ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(None) \
                         if df['Date'].dt.tz is not None \
                         else pd.to_datetime(df['Date'])
            keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                         'Dividends', 'Stock Splits', 'Ticker']
            for col in keep_cols:
                if col not in df.columns:
                    df[col] = 0
            new_records.append(df[keep_cols])
        except Exception:
            failed.append(ticker)

    if not new_records:
        print("❌ No new market data downloaded. Check internet connection.")
        return

    new_df = pd.concat(new_records, ignore_index=True)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Ticker', 'Date']).sort_values(['Ticker', 'Date'])

    backup = market_file.parent / 'stock_prices_backup_pre2024.csv'
    if not backup.exists():
        existing_df.to_csv(backup, index=False)
        print(f"   💾 Backup saved: {backup.name}")

    combined.to_csv(market_file, index=False)
    print(f"   ✅ {len(new_df):,} new records added")
    print(f"   📅 New range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    if failed:
        print(f"   ⚠️  {len(failed)} tickers failed: {failed[:5]}")
    print()

def generate_aligned_transcripts():
    """
    Generate sample transcripts that match actual financial data
    """
    print("="*60)
    print("PHASE 1D: GENERATING ALIGNED TRANSCRIPTS")
    print("="*60)
    print()
    
    # 1. Load financial data to get available tickers and dates
    financial_file = RAW_DATA_DIR / 'financial' / 'financial_features.csv'
    
    if not financial_file.exists():
        print(f"❌ Financial data not found: {financial_file}")
        print("   Please run Phase 1C first")
        return False
    
    print(f"📂 Reading financial data...")
    financials_df = pd.read_csv(financial_file)
    
    print(f"✅ Loaded {len(financials_df)} financial records")
    print(f"   Tickers: {financials_df['Ticker'].nunique()}")
    print()
    
    # 2. Get top 50 tickers with most data
    ticker_counts = financials_df['Ticker'].value_counts()
    top_50_tickers = ticker_counts.head(50).index.tolist()
    
    print(f"📊 Using top 50 tickers:")
    print(f"   {', '.join(top_50_tickers[:10])}...")
    print()
    
    # 3. Generate 3 transcripts per ticker (150 total)
    quarters = ['Q2 2024', 'Q3 2024', 'Q4 2024']
    dates = ['2024-07-31', '2024-10-31', '2025-01-31']
    
    transcripts = []
    
    for ticker in top_50_tickers:
        for i, (quarter, date) in enumerate(zip(quarters, dates)):
            transcript = {
                'ticker': ticker,
                'company_name': f"{ticker} Inc.",
                'quarter': quarter,
                'date': date,
                'fiscal_year': 2024,
                'fiscal_quarter': i + 2,  # Q2, Q3, Q4
                
                'prepared_remarks': f"Good afternoon, and thank you for joining {ticker}'s {quarter} earnings call. We delivered strong results this quarter with revenue growth across key segments. Our operational efficiency continues to improve, and we're well-positioned for the remainder of the year.",
                
                'qa_section': f"Analyst: Can you provide more details on the revenue growth?\n\nManagement: Thanks for the question. The revenue growth was driven by multiple factors including strong customer demand and operational improvements. We're seeing positive trends across all business segments.",
                
                'full_text': '',
                'word_count': 0,
                'speaker_count': 2,
                'source': 'SAMPLE_ALIGNED',
                'collection_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Combine full text
            transcript['full_text'] = transcript['prepared_remarks'] + "\n\n" + transcript['qa_section']
            transcript['word_count'] = len(transcript['full_text'].split())
            
            transcripts.append(transcript)
    
    print(f"✅ Generated {len(transcripts)} transcripts")
    print(f"   Tickers: {len(set(t['ticker'] for t in transcripts))}")
    print(f"   Quarters: {set(t['quarter'] for t in transcripts)}")
    print()
    
    # 4. Save transcripts
    output_dir = RAW_DATA_DIR / 'transcripts'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_df = pd.DataFrame([{
        'ticker': t['ticker'],
        'company_name': t['company_name'],
        'quarter': t['quarter'],
        'date': t['date'],
        'fiscal_year': t['fiscal_year'],
        'fiscal_quarter': t['fiscal_quarter'],
        'word_count': t['word_count'],
        'source': t['source'],
        'collection_date': t['collection_date']
    } for t in transcripts])
    
    metadata_path = output_dir / 'transcripts_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Save full JSON
    json_path = output_dir / 'transcripts_full.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    print(f"✅ Full transcripts saved: {json_path}")
    
    # Save individual files
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)
    
    for transcript in transcripts:
        filename = f"{transcript['ticker']}_{transcript['quarter'].replace(' ', '_')}.txt"
        file_path = individual_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Company: {transcript['company_name']}\n")
            f.write(f"Ticker: {transcript['ticker']}\n")
            f.write(f"Quarter: {transcript['quarter']}\n")
            f.write(f"Date: {transcript['date']}\n")
            f.write("="*60 + "\n\n")
            f.write(transcript['full_text'])
    
    print(f"✅ Individual files saved: {individual_dir}/")
    print()
    
    print("="*60)
    print("✅ PHASE 1D COMPLETE")
    print("="*60)
    print()
    print("📊 Summary:")
    print(f"   Total transcripts: {len(transcripts)}")
    print(f"   Unique tickers: {len(set(t['ticker'] for t in transcripts))}")
    print(f"   Date range: {min(t['date'] for t in transcripts)} to {max(t['date'] for t in transcripts)}")
    print()
    print("✅ Transcripts are now aligned with financial data!")
    print()
    
    # Update market data to cover 2024-2025
    update_market_data()
    
    return True

if __name__ == "__main__":
    success = generate_aligned_transcripts()
    
    if success:
        print("Next steps:")
        print("1. Re-run Phase 2A: python run_phase2a.py")
        print("2. Re-run Phase 2B: python run_phase2b.py")  
        print("3. Run Phase 2D: python run_phase2d.py")
        print("\nPhase 2D should now succeed!")