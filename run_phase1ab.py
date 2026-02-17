"""
Phase 1A & 1B Runner
Executes S&P 500 ticker collection and market data collection
"""

from src.data_collection.sp500_tickers import SP500TickerCollector
from src.data_collection.market_data import MarketDataCollector
from config import RAW_DATA_DIR

def run_phase_1ab():
    """
    Execute Phase 1A and 1B
    """
    print("\n" + "="*60)
    print("STARTING PHASE 1A & 1B")
    print("="*60)
    print()
    
    # Phase 1A: Collect tickers
    print("STEP 1: Collecting S&P 500 Tickers...")
    print()
    
    ticker_collector = SP500TickerCollector()
    tickers_df = ticker_collector.run()
    
    if tickers_df is None:
        print("\n❌ Phase 1A failed. Cannot proceed to Phase 1B.")
        return False
    
    print("\n" + "="*60)
    print()
    
    # Phase 1B: Collect market data
    print("STEP 2: Collecting Market Data...")
    print()
    
    tickers = tickers_df['Ticker'].tolist()
    market_collector = MarketDataCollector(tickers)
    stock_data, sp500_data = market_collector.run()
    
    if stock_data is None:
        print("\n❌ Phase 1B failed.")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PHASE 1A & 1B COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print(f"   1. S&P 500 tickers: {len(tickers_df)} companies")
    print(f"   2. Stock price data: {len(stock_data):,} records")
    print(f"   3. S&P 500 index data: {len(sp500_data) if sp500_data is not None else 0} records")
    print()
    print("📁 Data location:", RAW_DATA_DIR / 'market')
    print()
    print("✅ Ready for Phase 1C: Financial Statements Collection")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_1ab()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Waiting for your confirmation")
        print("="*60)
        print("\nPlease verify:")
        print("1. Check the data/raw/market/ folder")
        print("2. Confirm files are created")
        print("3. Review the statistics")
        print("\nWhen ready, confirm to proceed to Phase 1C")