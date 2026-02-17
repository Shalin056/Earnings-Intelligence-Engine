"""
Creates small sample datasets for Git repository
This allows others to test the code without full data
"""

import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR

def create_samples():
    """Create small sample datasets"""
    
    print("Creating sample datasets for Git...")
    
    # Sample tickers
    sample_dir = Path('data/samples')
    sample_dir.mkdir(exist_ok=True)
    
    # 1. Sample tickers (10 companies)
    tickers_df = pd.read_csv(RAW_DATA_DIR / 'market' / 'sp500_tickers.csv')
    sample_tickers = tickers_df.head(10)
    sample_tickers.to_csv(sample_dir / 'sample_tickers.csv', index=False)
    print("✅ Sample tickers created")
    
    # 2. Sample market data (100 rows)
    market_df = pd.read_csv(RAW_DATA_DIR / 'market' / 'stock_prices.csv')
    sample_market = market_df.head(100)
    sample_market.to_csv(sample_dir / 'sample_market_data.csv', index=False)
    print("✅ Sample market data created")
    
    # 3. Sample financials (50 rows)
    financial_df = pd.read_csv(RAW_DATA_DIR / 'financial' / 'financial_features.csv')
    sample_financials = financial_df.head(50)
    sample_financials.to_csv(sample_dir / 'sample_financials.csv', index=False)
    print("✅ Sample financials created")
    
    # 4. Sample transcript (1 file)
    import json
    with open(RAW_DATA_DIR / 'transcripts' / 'transcripts_full.json', 'r') as f:
        transcripts = json.load(f)
    
    sample_transcript = [transcripts[0]]
    with open(sample_dir / 'sample_transcript.json', 'w') as f:
        json.dump(sample_transcript, f, indent=2)
    print("✅ Sample transcript created")
    
    print(f"\n✅ All samples created in: {sample_dir}")
    print("These small files can be committed to Git for testing")

if __name__ == "__main__":
    create_samples()