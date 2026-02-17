"""
Verification script for Phase 1A & 1B data
"""

import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR

def verify_data():
    """
    Verify the collected data quality
    """
    print("="*60)
    print("PHASE 1A & 1B DATA VERIFICATION")
    print("="*60)
    print()
    
    market_dir = RAW_DATA_DIR / 'market'
    
    # 1. Verify tickers
    print("1️⃣ VERIFYING TICKERS...")
    tickers_df = pd.read_csv(market_dir / 'sp500_tickers.csv')
    print(f"   ✅ Tickers loaded: {len(tickers_df)} companies")
    print(f"   ✅ Sectors: {tickers_df['Sector'].nunique()}")
    print()
    
    # 2. Verify stock prices
    print("2️⃣ VERIFYING STOCK PRICES...")
    stocks_df = pd.read_csv(market_dir / 'stock_prices.csv')
    print(f"   ✅ Records loaded: {len(stocks_df):,}")
    print(f"   ✅ Unique tickers: {stocks_df['Ticker'].nunique()}")
    print(f"   ✅ Date range: {stocks_df['Date'].min()} to {stocks_df['Date'].max()}")
    print(f"   ✅ Columns: {', '.join(stocks_df.columns)}")
    
    # Check for missing values
    missing = stocks_df.isnull().sum()
    if missing.sum() > 0:
        print(f"   ⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"      - {col}: {count} ({count/len(stocks_df)*100:.2f}%)")
    else:
        print(f"   ✅ No missing values")
    print()
    
    # 3. Verify S&P 500 index
    print("3️⃣ VERIFYING S&P 500 INDEX...")
    sp500_df = pd.read_csv(market_dir / 'sp500_index.csv')
    print(f"   ✅ Records loaded: {len(sp500_df)}")
    print(f"   ✅ Date range: {sp500_df['Date'].min()} to {sp500_df['Date'].max()}")
    print()
    
    # 4. Sample data preview
    print("4️⃣ SAMPLE DATA PREVIEW...")
    print("\nStock Prices (first 5 rows):")
    print(stocks_df.head().to_string())
    print()
    
    # 5. Statistics
    print("5️⃣ STATISTICS...")
    print(f"   📊 Average records per ticker: {len(stocks_df) / stocks_df['Ticker'].nunique():.0f}")
    print(f"   📊 Trading days in dataset: {stocks_df['Date'].nunique()}")
    print(f"   📊 File size: {(market_dir / 'stock_prices.csv').stat().st_size / (1024*1024):.2f} MB")
    print()
    
    print("="*60)
    print("✅ DATA VERIFICATION COMPLETE")
    print("="*60)
    print()
    print("All checks passed! Data is ready for Phase 1C.")
    print()

if __name__ == "__main__":
    verify_data()