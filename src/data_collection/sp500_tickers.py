"""
Phase 1A: S&P 500 Ticker Collection
Collects the list of S&P 500 companies for analysis
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import sys
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, LOGS_DIR

class SP500TickerCollector:
    """
    Collects and manages S&P 500 ticker symbols
    """
    
    def __init__(self):
        self.output_dir = RAW_DATA_DIR / 'market'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'data_collection' / 'sp500_tickers.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def collect_tickers(self):
        """
        Scrapes S&P 500 tickers from Wikipedia with proper headers
        """
        print("="*60)
        print("PHASE 1A: S&P 500 TICKER COLLECTION")
        print("="*60)
        print()
        
        try:
            # Wikipedia URL
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            
            print(f"📡 Fetching data from Wikipedia...")
            
            # Headers to avoid 403 error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Get the page content
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse tables from HTML
            tables = pd.read_html(StringIO(response.text))
            
            # First table contains current S&P 500 companies
            sp500_table = tables[0]
            
            print(f"✅ Successfully retrieved {len(sp500_table)} companies")
            print()
            
            # Extract relevant columns
            tickers_df = sp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 
                                       'Headquarters Location', 'Date added', 'Founded']].copy()
            
            # Rename columns for clarity
            tickers_df.columns = ['Ticker', 'Company_Name', 'Sector', 'Sub_Industry', 
                                  'Headquarters', 'Date_Added', 'Founded']
            
            # Clean ticker symbols (some have periods, dots etc)
            tickers_df['Ticker'] = tickers_df['Ticker'].str.replace('.', '-')
            
            # Add collection metadata
            tickers_df['Collection_Date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Display sample
            print("📊 Sample of collected tickers:")
            print(tickers_df.head(10).to_string())
            print()
            
            # Statistics
            print("📈 Statistics:")
            print(f"   Total companies: {len(tickers_df)}")
            print(f"   Sectors: {tickers_df['Sector'].nunique()}")
            print()
            
            # Sector breakdown
            print("🏢 Sector Breakdown:")
            sector_counts = tickers_df['Sector'].value_counts()
            for sector, count in sector_counts.items():
                print(f"   {sector}: {count}")
            print()
            
            return tickers_df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error collecting tickers: {e}")
            print("\n⚠️  Attempting fallback method...")
            return self._fallback_ticker_collection()
            
        except Exception as e:
            print(f"❌ Error collecting tickers: {e}")
            print("\n⚠️  Attempting fallback method...")
            return self._fallback_ticker_collection()
    
    def _fallback_ticker_collection(self):
        """
        Fallback method: Use a pre-defined list of S&P 500 tickers
        This ensures the project can continue even if Wikipedia is blocked
        """
        print("📋 Using fallback ticker list (top 100 S&P 500 companies)...")
        
        # Top 100 most liquid S&P 500 stocks (as of 2024)
        fallback_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'XOM', 'V', 'JPM', 'PG', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV',
            'PEP', 'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
            'BAC', 'CRM', 'ADBE', 'DHR', 'LIN', 'NFLX', 'VZ', 'NKE', 'TXN', 'CMCSA',
            'WFC', 'NEE', 'DIS', 'RTX', 'ORCL', 'AMD', 'UPS', 'PM', 'QCOM', 'MS',
            'INTU', 'HON', 'T', 'IBM', 'AMGN', 'COP', 'LOW', 'AMAT', 'GE', 'CAT',
            'SPGI', 'UNP', 'PLD', 'NOW', 'GS', 'SBUX', 'DE', 'BLK', 'ELV', 'AXP',
            'BKNG', 'BA', 'SYK', 'GILD', 'ADI', 'TJX', 'MDLZ', 'VRTX', 'MMC', 'C',
            'ADP', 'ISRG', 'PGR', 'CI', 'LRCX', 'REGN', 'MO', 'ZTS', 'CVS', 'CB',
            'SO', 'TMUS', 'PNC', 'DUK', 'BMY', 'EOG', 'TGT', 'HUM', 'USB', 'ITW'
        ]
        
        # Create DataFrame with basic info
        tickers_df = pd.DataFrame({
            'Ticker': fallback_tickers,
            'Company_Name': ['Company ' + ticker for ticker in fallback_tickers],
            'Sector': 'Various',
            'Sub_Industry': 'Various',
            'Headquarters': 'United States',
            'Date_Added': 'N/A',
            'Founded': 'N/A',
            'Collection_Date': datetime.now().strftime('%Y-%m-%d'),
            'Source': 'Fallback List'
        })
        
        print(f"✅ Loaded {len(tickers_df)} tickers from fallback list")
        print("\n⚠️  Note: Using reduced ticker set. You can manually add more tickers later.")
        print()
        
        return tickers_df
    
    def save_tickers(self, tickers_df):
        """
        Save tickers to CSV
        """
        if tickers_df is None:
            print("❌ No data to save")
            return False
        
        try:
            # Save full dataset
            output_path = self.output_dir / 'sp500_tickers.csv'
            tickers_df.to_csv(output_path, index=False)
            print(f"✅ Saved to: {output_path}")
            
            # Save just ticker list for convenience
            ticker_list_path = self.output_dir / 'sp500_ticker_list.txt'
            tickers_df['Ticker'].to_csv(ticker_list_path, index=False, header=False)
            print(f"✅ Saved ticker list to: {ticker_list_path}")
            
            # Log success
            self._log_success(len(tickers_df))
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving tickers: {e}")
            self._log_error(f"Save error: {e}")
            return False
    
    def _log_success(self, count):
        """Log successful collection"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - SUCCESS: Collected {count} tickers\n")
    
    def _log_error(self, error_msg):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {error_msg}\n")
    
    def run(self):
        """
        Execute complete ticker collection workflow
        """
        tickers_df = self.collect_tickers()
        
        if tickers_df is not None:
            success = self.save_tickers(tickers_df)
            
            if success:
                print("\n" + "="*60)
                print("✅ PHASE 1A COMPLETE")
                print("="*60)
                print(f"\n📁 Data saved to: {self.output_dir}")
                print(f"📝 Log saved to: {self.log_file}")
                print("\n✅ Ready for Phase 1B: Market Data Collection")
                return tickers_df
        
        return None

def main():
    """Main execution"""
    collector = SP500TickerCollector()
    tickers_df = collector.run()
    return tickers_df

if __name__ == "__main__":
    main()