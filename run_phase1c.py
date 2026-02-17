"""
Phase 1C Runner
Executes Financial Statements Collection
"""

from src.data_collection.financial_statements import FinancialStatementsCollector
from config import RAW_DATA_DIR
import pandas as pd

def run_phase_1c():
    """
    Execute Phase 1C
    """
    print("\n" + "="*60)
    print("STARTING PHASE 1C")
    print("="*60)
    print()
    
    # Load tickers
    ticker_path = RAW_DATA_DIR / 'market' / 'sp500_tickers.csv'
    
    if not ticker_path.exists():
        print("❌ Ticker file not found. Please run Phase 1A first.")
        return False
    
    tickers_df = pd.read_csv(ticker_path)
    tickers = tickers_df['Ticker'].tolist()
    
    print(f"📊 Loaded {len(tickers)} tickers")
    print()
    
    # Collect financial statements
    collector = FinancialStatementsCollector(tickers)
    income, balance, cashflow, features = collector.run()
    
    if income is None:
        print("\n❌ Phase 1C failed.")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PHASE 1C COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print(f"   1. Income Statements: {len(income):,} records")
    print(f"   2. Balance Sheets: {len(balance):,} records")
    print(f"   3. Cash Flows: {len(cashflow):,} records")
    print(f"   4. Financial Features: {len(features):,} records with 28 features")
    print()
    print("📁 Data location:", RAW_DATA_DIR / 'financial')
    print()
    print("✅ Ready for Phase 1D: Earnings Call Transcripts")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_1c()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Waiting for your confirmation")
        print("="*60)
        print("\nPlease verify:")
        print("1. Check the data/raw/financial/ folder")
        print("2. Confirm files are created")
        print("3. Review the feature completeness")
        print("\nWhen ready, confirm to proceed to Phase 1D")