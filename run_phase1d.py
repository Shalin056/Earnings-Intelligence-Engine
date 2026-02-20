# """
# Phase 1D Fixed - Generate transcripts matching actual financial data
# """

# import pandas as pd
# import json
# from pathlib import Path
# from datetime import datetime
# from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# def update_market_data():
#     """Update market data to cover 2024-2025 transcript date range."""
#     import yfinance as yf
#     from tqdm import tqdm

#     market_file = RAW_DATA_DIR / 'market' / 'stock_prices.csv'
#     if not market_file.exists():
#         print("⚠️  Market data file not found, skipping update")
#         return

#     print("="*60)
#     print("📡 UPDATING MARKET DATA TO 2024-2025")
#     print("="*60)

#     existing_df = pd.read_csv(market_file)
#     existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True).dt.tz_convert(None)
#     current_end = existing_df['Date'].max()
#     print(f"   Current data ends: {current_end.date()}")

#     if current_end >= pd.Timestamp('2025-01-01'):
#         print("   ✅ Market data already up to date, skipping")
#         return

#     tickers = existing_df['Ticker'].unique().tolist()
#     START_DATE = (current_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
#     END_DATE = "2026-02-20"
#     print(f"   Downloading: {START_DATE} to {END_DATE} for {len(tickers)} tickers")

#     new_records = []
#     failed = []
#     for ticker in tqdm(tickers, desc="Updating market data"):
#         try:
#             df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=False)
#             if df.empty:
#                 failed.append(ticker)
#                 continue
#             df = df.reset_index()
#             df['Ticker'] = ticker
#             df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(None) \
#                          if df['Date'].dt.tz is not None \
#                          else pd.to_datetime(df['Date'])
#             keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
#                          'Dividends', 'Stock Splits', 'Ticker']
#             for col in keep_cols:
#                 if col not in df.columns:
#                     df[col] = 0
#             new_records.append(df[keep_cols])
#         except Exception:
#             failed.append(ticker)

#     if not new_records:
#         print("❌ No new market data downloaded. Check internet connection.")
#         return

#     new_df = pd.concat(new_records, ignore_index=True)
#     combined = pd.concat([existing_df, new_df], ignore_index=True)
#     combined = combined.drop_duplicates(subset=['Ticker', 'Date']).sort_values(['Ticker', 'Date'])

#     backup = market_file.parent / 'stock_prices_backup_pre2024.csv'
#     if not backup.exists():
#         existing_df.to_csv(backup, index=False)
#         print(f"   💾 Backup saved: {backup.name}")

#     combined.to_csv(market_file, index=False)
#     print(f"   ✅ {len(new_df):,} new records added")
#     print(f"   📅 New range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
#     if failed:
#         print(f"   ⚠️  {len(failed)} tickers failed: {failed[:5]}")
#     print()

# def generate_aligned_transcripts():
#     """
#     Generate sample transcripts that match actual financial data
#     """
#     print("="*60)
#     print("PHASE 1D: GENERATING ALIGNED TRANSCRIPTS")
#     print("="*60)
#     print()
    
#     # 1. Load financial data to get available tickers and dates
#     financial_file = RAW_DATA_DIR / 'financial' / 'financial_features.csv'
    
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
    
#     # Update market data to cover 2024-2025
#     update_market_data()
    
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
Phase 1D - Generate aligned transcripts (real + synthetic)
"""

import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def update_market_data():
    """Update market data to cover 2024-2026 transcript date range."""
    import yfinance as yf
    from tqdm import tqdm

    market_file = RAW_DATA_DIR / 'market' / 'stock_prices.csv'
    if not market_file.exists():
        print("⚠️  Market data file not found, skipping update")
        return

    print("="*60)
    print("📡 UPDATING MARKET DATA TO 2024-2026")
    print("="*60)

    existing_df = pd.read_csv(market_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True).dt.tz_convert(None)
    current_end = existing_df['Date'].max()
    print(f"   Current data ends: {current_end.date()}")

    if current_end >= pd.Timestamp('2026-01-01'):
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


def _make_remarks(ticker, quarter):
    rng = random.Random(hash(ticker + quarter))
    rev = rng.choice(['$4.2 billion', '$7.8 billion', '$2.1 billion', '$12.4 billion', '$890 million', '$3.6 billion'])
    rev_growth = rng.choice(['+3%', '+8%', '+14%', '+2%', '-1%', '+22%', '+6%', '-4%'])
    margin = rng.choice(['18.2%', '24.7%', '31.1%', '12.4%', '8.9%', '42.3%', '19.8%'])
    eps = rng.choice(['$1.24', '$3.87', '$0.94', '$2.15', '$5.32', '$0.67', '$1.89'])
    metrics = rng.sample(['cloud adoption', 'AI integration', 'net interest margin', 'loan growth',
                           'drug pipeline', 'same-store sales', 'production volumes', 'backlog growth',
                           'software subscriptions', 'fee income', 'digital commerce', 'pricing realization'], 2)
    m1, m2 = metrics[0], metrics[1]
    scenario = rng.randint(0, 5)

    if scenario == 0:
        return f"""Good morning, and thank you for joining {ticker}'s {quarter} earnings call.

I am thrilled to report an exceptional quarter that significantly exceeded our expectations.
Revenue came in at {rev}, representing {rev_growth} year-over-year growth, beating consensus
estimates by a wide margin. Our {m1} business delivered record performance, and {m2}
continues to be a meaningful growth driver.

Operating margins expanded to {margin}, a 280 basis point improvement versus the prior year.
Diluted EPS of {eps} reflects the operational leverage in our business model. Free cash flow
generation was outstanding, and we returned significant capital to shareholders.

Customer demand is accelerating. We are seeing strong adoption across all geographies,
and our pipeline has never been stronger. We are raising full-year guidance and are
highly confident in our ability to deliver sustained, profitable growth."""

    elif scenario == 1:
        return f"""Good morning. Thank you for joining {ticker}'s {quarter} earnings call.

I want to be direct with you today. Our results for the quarter came in below our
expectations. Revenue of {rev} was {rev_growth} year-over-year, which fell short of both
our internal targets and consensus estimates.

We faced a challenging environment. {m1.capitalize()} deteriorated more rapidly than
anticipated, and the headwinds in {m2} proved more persistent than we modeled. Margins
compressed to {margin} due to elevated costs and pricing pressure we could not fully offset.

EPS of {eps} reflects these operational challenges. We are not satisfied with this
performance and are taking immediate action including cost reductions and leadership changes.
We are lowering our full-year guidance to reflect current market realities."""

    elif scenario == 2:
        return f"""Thank you for joining us today to discuss {ticker}'s {quarter} results.

Our performance this quarter was mixed. Revenue of {rev} grew {rev_growth} year-over-year,
in line with expectations, though we saw meaningful divergence across business segments.
Our {m1} business performed well, while {m2} faced headwinds we are actively managing.

Operating margins of {margin} were roughly flat sequentially. EPS of {eps} reflects
disciplined cost management partially offsetting top-line softness.

We are maintaining our full-year guidance range, though we acknowledge the range of
outcomes has widened. Our balance sheet is strong, which gives us flexibility."""

    elif scenario == 3:
        return f"""Good morning, and thank you for joining {ticker}'s {quarter} earnings call.

I am pleased to report that our turnaround is gaining momentum. Revenue of {rev}
represents {rev_growth} growth — the third consecutive quarter of sequential improvement.
After a difficult period, the strategic actions we took are beginning to yield results.

Our restructuring is substantially complete and we now operate with a leaner cost structure.
Margins recovered to {margin}. EPS of {eps} demonstrates that profitability is returning.
We are reinstating guidance and see a credible path to returning to long-term targets."""

    elif scenario == 4:
        return f"""Thank you for joining {ticker}'s {quarter} earnings call.

We delivered another strong quarter of top-line growth with revenue of {rev}, up {rev_growth}
year-over-year. Our {m1} segment is scaling exceptionally well and {m2} is becoming
increasingly meaningful. We are intentionally investing ahead of the growth curve, which
is why margins of {margin} reflect higher operating expenses. EPS of {eps} reflects this
deliberate investment posture. We are raising our revenue growth outlook for the full year."""

    else:
        return f"""Good morning. Thank you for joining us for {ticker}'s {quarter} earnings call.

We operated in a challenging macro environment this quarter. Revenue of {rev} was {rev_growth}
versus the prior year period. We are seeing a more cautious purchasing environment, particularly
in {m1}. Margins held at {margin} through cost discipline. EPS of {eps} reflects these
efficiency efforts. We are widening our guidance range to reflect macro uncertainty."""


def _make_qa(ticker, quarter):
    rng = random.Random(hash(ticker + quarter + "qa"))
    scenario = rng.randint(0, 4)

    if scenario == 0:
        return f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Goldman Sachs): I want to push back. The revenue miss was meaningful and
guidance implies continued deceleration. What specifically went wrong?

Management: The primary driver was elongated sales cycles in enterprise — deals slipped
into next quarter. Underlying demand has not deteriorated. We have made structural changes
to our sales process including restructured compensation to penalize slippage.

Analyst 2 (Morgan Stanley): SG&A went up as a percentage of revenue. Can you walk us through that?

Management: We had one-time items — severance from reorganization and legal accruals.
Stripping those out, core SG&A was down year-over-year. We target mid-twenties operating
margin within four to six quarters."""

    elif scenario == 1:
        return f"""
QUESTION AND ANSWER SESSION

Analyst 1 (JPMorgan): Fantastic quarter. The beat was broad-based. What drove the upside?

Management: Three things came together. Go-to-market changes now in full effect with stronger
attach rates. International markets accelerated, particularly EMEA. Gross margins benefited
from favorable mix shift we expect to persist.

Analyst 2 (Bank of America): You have built up substantial cash. What is the priority use?

Management: Organic investment remains priority one. Our internal pipeline returns are
excellent. Beyond that we consider tuck-in acquisitions and continue buybacks. We are
very disciplined on M&A and have passed on several deals that did not meet return thresholds."""

    elif scenario == 2:
        return f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Citi): I am trying to understand sustainability of margins. What gives you confidence?

Management: We have permanently taken out structural costs — roughly 150 basis points of
permanent savings. Pricing actions from earlier this year are now fully in the run rate.
We also have a variable cost structure that allows us to flex down quickly if needed.

Analyst 2 (Wells Fargo): There has been aggressive pricing from a key competitor.
How are win rates holding?

Management: Win rates have remained stable, which tells us customers value our differentiation.
We will not race to the bottom on price. Our value proposition commands a premium."""

    elif scenario == 3:
        return f"""
QUESTION AND ANSWER SESSION

Analyst 1 (UBS): The guidance midpoint implies a meaningful step-down in growth.
Can you bridge that?

Management: We have a tougher year-over-year comparison in the back half from a large
one-time government contract last year that will not repeat — roughly $180 million above
our typical run rate. We are also being more conservative on macro.

Analyst 2 (Deutsche Bank): Debt levels are elevated. What is the path to deleveraging?

Management: We are committed to getting back to target leverage of 2 to 2.5 times within
18 months. Free cash flow is our primary tool — we are prioritizing debt paydown over
buybacks for the next several quarters."""

    else:
        return f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Barclays): Three questions. Revenue run rate exiting the year? Margin floor?
When does free cash flow conversion normalize?

Management: Revenue exit rate — sequential improvement each quarter, implying Q4 exit
rate of $1.1 to $1.2 billion annualized. Margin floor — we believe we have seen it, cost
actions create a floor in high single digits. Free cash flow — working capital normalization
drives conversion back above 90% by Q4.

Analyst 2 (Bernstein): Stock is down significantly. Does management feel urgency around
capital return?

Management: We are acutely aware. The board authorized an incremental $500 million buyback
and we intend to be active in the open market. We think the stock is meaningfully undervalued."""


def generate_aligned_transcripts():
    """
    Load real transcripts (>=20KB) first, then fill with synthetic ones.
    All transcripts aligned with financial and market data.
    """
    print("="*60)
    print("PHASE 1D: GENERATING ALIGNED TRANSCRIPTS")
    print("="*60)
    print()

    # 1. Load financial data
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

    # Quarter -> earnings call date mapping
    QUARTER_TO_DATE = {
        'Q1 2024': '2024-04-30', 'Q2 2024': '2024-07-31',
        'Q3 2024': '2024-10-31', 'Q4 2024': '2025-01-31',
        'Q1 2025': '2025-04-30', 'Q2 2025': '2025-07-31',
        'Q3 2025': '2025-10-31', 'Q4 2025': '2026-01-31',
    }
    MARKET_DATA_END = '2026-02-19'

    # 2. Load real transcripts (>=20KB)
    individual_dir = RAW_DATA_DIR / 'transcripts' / 'individual'
    transcripts = []
    real_keys = set()

    if individual_dir.exists():
        print("📂 Scanning for real transcripts (>=20KB)...")
        for f in sorted(individual_dir.glob('*.txt')):
            if f.stat().st_size < 20_000:
                continue
            parts = f.stem.split('_')
            if len(parts) < 3:
                continue
            ticker = parts[0]
            q_num  = parts[1]
            year   = parts[2]
            quarter = f"{q_num} {year}"
            date = QUARTER_TO_DATE.get(quarter)
            if date is None or date > MARKET_DATA_END:
                print(f"   ⚠️  Skipping {f.name} — date outside market range")
                continue
            full_text = f.read_text(encoding='utf-8', errors='ignore')
            transcript = {
                'ticker': ticker,
                'company_name': f"{ticker} Inc.",
                'quarter': quarter,
                'date': date,
                'fiscal_year': int(year),
                'fiscal_quarter': int(q_num[1]),
                'prepared_remarks': full_text,
                'qa_section': '',
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'speaker_count': 2,
                'source': 'REAL',
                'collection_date': datetime.now().strftime('%Y-%m-%d')
            }
            transcripts.append(transcript)
            real_keys.add((ticker, quarter))

        print(f"✅ Loaded {len(transcripts)} real transcripts")
        print()

    # 3. Fill with synthetic transcripts for remaining tickers/quarters
    print(f"🤖 Generating synthetic transcripts...")
    ticker_counts = financials_df['Ticker'].value_counts()
    all_tickers = ticker_counts.index.tolist()

    synthetic_quarters = ['Q2 2024', 'Q3 2024', 'Q4 2024']
    synthetic_dates    = ['2024-07-31', '2024-10-31', '2025-01-31']

    for ticker in all_tickers:
        for quarter, date in zip(synthetic_quarters, synthetic_dates):
            if (ticker, quarter) in real_keys:
                continue  # already have real transcript for this
            prepared = _make_remarks(ticker, quarter)
            qa = _make_qa(ticker, quarter)
            full_text = prepared + "\n\n" + qa
            transcripts.append({
                'ticker': ticker,
                'company_name': f"{ticker} Inc.",
                'quarter': quarter,
                'date': date,
                'fiscal_year': 2024,
                'fiscal_quarter': synthetic_quarters.index(quarter) + 2,
                'prepared_remarks': prepared,
                'qa_section': qa,
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'speaker_count': 2,
                'source': 'SYNTHETIC',
                'collection_date': datetime.now().strftime('%Y-%m-%d')
            })

    real_count = sum(1 for t in transcripts if t['source'] == 'REAL')
    synth_count = sum(1 for t in transcripts if t['source'] == 'SYNTHETIC')
    print(f"✅ Total transcripts: {len(transcripts)}")
    print(f"   Real: {real_count}")
    print(f"   Synthetic: {synth_count}")
    print(f"   Unique tickers: {len(set(t['ticker'] for t in transcripts))}")
    print()

    # 4. Save
    output_dir = RAW_DATA_DIR / 'transcripts'
    output_dir.mkdir(parents=True, exist_ok=True)

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

    json_path = output_dir / 'transcripts_full.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    print(f"✅ Full transcripts saved: {json_path}")

    individual_dir_out = output_dir / 'individual'
    individual_dir_out.mkdir(exist_ok=True)
    for transcript in transcripts:
        filename = f"{transcript['ticker']}_{transcript['quarter'].replace(' ', '_')}.txt"
        file_path = individual_dir_out / filename
        if transcript['source'] == 'REAL':
            continue  # don't overwrite real transcripts
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Company: {transcript['company_name']}\n")
            f.write(f"Ticker: {transcript['ticker']}\n")
            f.write(f"Quarter: {transcript['quarter']}\n")
            f.write(f"Date: {transcript['date']}\n")
            f.write("="*60 + "\n\n")
            f.write(transcript['full_text'])

    print(f"✅ Individual files saved: {individual_dir_out}/")
    print()

    print("="*60)
    print("✅ PHASE 1D COMPLETE")
    print("="*60)
    print()
    print("📊 Summary:")
    print(f"   Total transcripts: {len(transcripts)}")
    print(f"   Real transcripts: {real_count}")
    print(f"   Synthetic transcripts: {synth_count}")
    print(f"   Unique tickers: {len(set(t['ticker'] for t in transcripts))}")
    print(f"   Date range: {min(t['date'] for t in transcripts)} to {max(t['date'] for t in transcripts)}")
    print()
    print("✅ Transcripts are now aligned with financial data!")
    print()

    update_market_data()

    return True


if __name__ == "__main__":
    success = generate_aligned_transcripts()
    if success:
        print("Next steps:")
        print("1. Re-run Phase 2A: python run_phase2a.py")
        print("2. Re-run Phase 2B: python run_phase2b.py")
        print("3. Run Phase 2C: python run_phase2c.py")
        print("4. Run Phase 2D: python run_phase2d.py")