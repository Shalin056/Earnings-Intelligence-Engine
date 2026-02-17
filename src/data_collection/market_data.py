"""
Phase 1B: Stock Market Data Collection
Collects daily OHLCV data for S&P 500 companies using yfinance
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, LOGS_DIR, START_DATE, END_DATE

class MarketDataCollector:
    """
    Collects daily stock price data from Yahoo Finance
    """
    
    def __init__(self, tickers, start_date=START_DATE, end_date=END_DATE):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        self.output_dir = RAW_DATA_DIR / 'market'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'data_collection' / 'market_data.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.failed_tickers = []
        self.successful_tickers = []
    
    def collect_stock_data(self, ticker):
        """
        Collect data for a single ticker
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                return None
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            return df
            
        except Exception as e:
            self._log_error(f"{ticker}: {e}")
            return None
    
    def collect_all_stocks(self):
        """
        Collect data for all tickers
        """
        print("="*60)
        print("PHASE 1B: MARKET DATA COLLECTION")
        print("="*60)
        print()
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"📊 Tickers to process: {len(self.tickers)}")
        print()
        
        all_data = []
        
        # Progress bar
        for ticker in tqdm(self.tickers, desc="Collecting market data"):
            df = self.collect_stock_data(ticker)
            
            if df is not None and not df.empty:
                all_data.append(df)
                self.successful_tickers.append(ticker)
            else:
                self.failed_tickers.append(ticker)
            
            # Rate limiting - be nice to Yahoo Finance
            time.sleep(0.1)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            print()
            print("="*60)
            print("📊 COLLECTION SUMMARY")
            print("="*60)
            print(f"✅ Successful: {len(self.successful_tickers)}/{len(self.tickers)}")
            print(f"❌ Failed: {len(self.failed_tickers)}/{len(self.tickers)}")
            print(f"📈 Total records: {len(combined_df):,}")
            print(f"📅 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
            print()
            
            if self.failed_tickers:
                print(f"⚠️  Failed tickers: {', '.join(self.failed_tickers[:10])}")
                if len(self.failed_tickers) > 10:
                    print(f"   ... and {len(self.failed_tickers) - 10} more")
                print()
            
            return combined_df
        
        return None
    
    def collect_sp500_index(self):
        """
        Collect S&P 500 index data (for abnormal returns calculation)
        """
        print("📊 Collecting S&P 500 index data (^GSPC)...")
        
        try:
            sp500 = yf.Ticker("^GSPC")
            df = sp500.history(start=self.start_date, end=self.end_date)
            
            if not df.empty:
                df['Ticker'] = 'SP500'
                df.reset_index(inplace=True)
                print(f"✅ S&P 500 data collected: {len(df)} records")
                return df
            else:
                print("❌ Failed to collect S&P 500 data")
                return None
                
        except Exception as e:
            print(f"❌ Error collecting S&P 500: {e}")
            return None
    
    def save_data(self, stock_data, sp500_data):
        """
        Save collected data to CSV
        """
        try:
            # Save individual stock data
            stock_path = self.output_dir / 'stock_prices.csv'
            stock_data.to_csv(stock_path, index=False)
            print(f"✅ Stock data saved to: {stock_path}")
            print(f"   Size: {stock_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save S&P 500 data
            if sp500_data is not None:
                sp500_path = self.output_dir / 'sp500_index.csv'
                sp500_data.to_csv(sp500_path, index=False)
                print(f"✅ S&P 500 data saved to: {sp500_path}")
            
            # Save success/failure log
            log_data = {
                'Successful_Tickers': self.successful_tickers,
                'Failed_Tickers': self.failed_tickers
            }
            log_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in log_data.items()]))
            log_path = self.output_dir / 'collection_log.csv'
            log_df.to_csv(log_path, index=False)
            print(f"✅ Collection log saved to: {log_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False
    
    def _log_error(self, error_msg):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - {error_msg}\n")
    
    def run(self):
        """
        Execute complete market data collection workflow
        """
        # Collect stock data
        stock_data = self.collect_all_stocks()
        
        # Collect S&P 500 index
        sp500_data = self.collect_sp500_index()
        
        if stock_data is not None:
            # Save data
            success = self.save_data(stock_data, sp500_data)
            
            if success:
                print("\n" + "="*60)
                print("✅ PHASE 1B COMPLETE")
                print("="*60)
                print(f"\n📁 Data saved to: {self.output_dir}")
                
                # Calculate success rate
                success_rate = len(self.successful_tickers) / len(self.tickers)
                print(f"\n📊 Success Rate: {success_rate:.1%}")
                
                if success_rate >= 0.90:
                    print("✅ Success rate meets threshold (≥90%)")
                else:
                    print("⚠️  Success rate below 90% threshold")
                
                return stock_data, sp500_data
        
        return None, None

def main():
    """Main execution"""
    # Load tickers
    ticker_path = RAW_DATA_DIR / 'market' / 'sp500_tickers.csv'
    
    if not ticker_path.exists():
        print("❌ Ticker file not found. Please run sp500_tickers.py first.")
        return
    
    tickers_df = pd.read_csv(ticker_path)
    tickers = tickers_df['Ticker'].tolist()
    
    # Collect market data
    collector = MarketDataCollector(tickers)
    stock_data, sp500_data = collector.run()
    
    return stock_data, sp500_data

if __name__ == "__main__":
    main()