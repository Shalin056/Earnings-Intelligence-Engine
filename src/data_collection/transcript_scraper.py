# """
# Phase 1D: Earnings Call Transcript Collection
# Collects earnings call transcripts from multiple sources
# """

# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from pathlib import Path
# from datetime import datetime
# import time
# import json
# import re
# import sys
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# # Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))

# from config import RAW_DATA_DIR, LOGS_DIR

# class TranscriptCollector:
#     """
#     Collects earnings call transcripts from multiple sources
#     """
    
#     def __init__(self, target_count=150):
#         self.target_count = target_count
        
#         self.output_dir = RAW_DATA_DIR / 'transcripts'
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.log_file = LOGS_DIR / 'data_collection' / 'transcripts.log'
#         self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
#         self.collected_transcripts = []
#         self.failed_attempts = []
        
#         # Headers to avoid blocking
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#             'Accept-Language': 'en-US,en;q=0.5',
#             'Accept-Encoding': 'gzip, deflate',
#             'Connection': 'keep-alive',
#             'Upgrade-Insecure-Requests': '1'
#         }
    
#     def get_high_priority_tickers(self):
#         """
#         Get tickers that ACTUALLY have financial data
#         This ensures alignment in Phase 2D
#         """
#         from config import PROCESSED_DATA_DIR
#         import pandas as pd
        
#         # Read tickers from actual financial data
#         financial_file = PROCESSED_DATA_DIR / 'financial' / 'financial_features_normalized.csv'
        
#         if financial_file.exists():
#             print("📊 Reading tickers from financial data...")
#             df = pd.read_csv(financial_file)
#             available_tickers = df['Ticker'].unique().tolist()
#             print(f"✅ Found {len(available_tickers)} tickers with financial data")
            
#             # Return first 50 tickers that have the most data
#             ticker_counts = df['Ticker'].value_counts()
#             top_tickers = ticker_counts.head(50).index.tolist()
            
#             return top_tickers
#         else:
#             print("⚠️  Financial data not found, using default tickers")
#             # Fallback to default list
#             return [
#                 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
#                 'XOM', 'V', 'JPM', 'PG', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV',
#                 'PEP', 'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
#                 'BAC', 'CRM', 'ADBE', 'DHR', 'LIN', 'NFLX', 'VZ', 'NKE', 'TXN', 'CMCSA',
#                 'WFC', 'NEE', 'DIS', 'RTX', 'ORCL', 'AMD', 'UPS', 'PM', 'QCOM', 'MS'
#             ]
    
#     def collect_from_sec_edgar(self, ticker, max_transcripts=3):
#         """
#         Collect transcripts from SEC EDGAR (8-K filings)
#         These often contain earnings call information
#         """
#         collected = []
        
#         try:
#             # SEC EDGAR search URL
#             # Note: This is a simplified version. Real implementation would use SEC API
#             url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=exclude&count=10"
            
#             print(f"   Attempting SEC EDGAR for {ticker}...")
            
#             # For now, we'll mark this as a placeholder
#             # Real implementation would parse SEC filings
#             self._log_info(f"SEC EDGAR: {ticker} - Placeholder (requires SEC API access)")
            
#             return collected
            
#         except Exception as e:
#             self._log_error(f"SEC EDGAR {ticker}: {e}")
#             return collected
    
#     def collect_from_seeking_alpha_sample(self, ticker, max_transcripts=3):
#         """
#         Attempt to collect transcripts from Seeking Alpha
#         CAUTIOUS approach with rate limiting
#         """
#         collected = []
        
#         try:
#             # Seeking Alpha earnings URL pattern
#             base_url = "https://seekingalpha.com/symbol"
#             url = f"{base_url}/{ticker}/earnings/transcripts"
            
#             print(f"   Attempting Seeking Alpha for {ticker}...")
            
#             # Make request with headers
#             response = requests.get(url, headers=self.headers, timeout=10)
            
#             # Check if blocked
#             if response.status_code == 403:
#                 print(f"   ⚠️  Blocked by Seeking Alpha for {ticker}")
#                 self._log_error(f"Seeking Alpha {ticker}: 403 Forbidden")
#                 return collected
            
#             if response.status_code != 200:
#                 print(f"   ⚠️  Failed to access {ticker}: Status {response.status_code}")
#                 return collected
            
#             # Parse HTML
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # This is simplified - actual parsing would need to match SA's HTML structure
#             # For demonstration, we'll create placeholder structure
#             print(f"   ⚠️  Seeking Alpha parsing requires authentication - using placeholder")
#             self._log_info(f"Seeking Alpha {ticker}: Requires authentication")
            
#             return collected
            
#         except Exception as e:
#             self._log_error(f"Seeking Alpha {ticker}: {e}")
#             return collected
    
#     def create_sample_transcripts(self, tickers, transcripts_per_ticker=3):
#         """
#         Create sample/demo transcripts for proof-of-concept
#         This allows the project to proceed while real transcripts are collected manually
#         """
#         print("="*60)
#         print("CREATING SAMPLE TRANSCRIPTS")
#         print("="*60)
#         print()
#         print("⚠️  Note: These are structured samples for pipeline development")
#         print("   Real transcripts should be added manually to data/raw/transcripts/")
#         print()
        
#         sample_transcripts = []
        
#         quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
        
#         for ticker in tqdm(tickers[:50], desc="Generating sample transcripts"):  # 50 tickers x 3 = 150
#             for i in range(min(transcripts_per_ticker, len(quarters))):
#                 quarter = quarters[i]
                
#                 # Create realistic sample transcript structure
#                 transcript = {
#                     'ticker': ticker,
#                     'company_name': f"{ticker} Inc.",
#                     'quarter': quarter,
#                     'date': self._get_quarter_date(quarter),
#                     'fiscal_year': 2023,
#                     'fiscal_quarter': i + 1,
                    
#                     # Prepared Remarks (Management)
#                     'prepared_remarks': self._generate_sample_prepared_remarks(ticker, quarter),
                    
#                     # Q&A Section
#                     'qa_section': self._generate_sample_qa(ticker, quarter),
                    
#                     # Full transcript
#                     'full_text': '',
                    
#                     # Metadata
#                     'word_count': 0,
#                     'speaker_count': 0,
#                     'source': 'SAMPLE',
#                     'collection_date': datetime.now().strftime('%Y-%m-%d')
#                 }
                
#                 # Combine sections
#                 transcript['full_text'] = transcript['prepared_remarks'] + "\n\n" + transcript['qa_section']
#                 transcript['word_count'] = len(transcript['full_text'].split())
                
#                 sample_transcripts.append(transcript)
        
#         print(f"\n✅ Generated {len(sample_transcripts)} sample transcripts")
#         print()
        
#         return sample_transcripts
    
#     def _generate_sample_prepared_remarks(self, ticker, quarter):
#         """Generate realistic sample prepared remarks"""
        
#         templates = [
#             f"""Good morning, and thank you for joining {ticker}'s {quarter} earnings call. 
            
# I'm pleased to report strong financial results for the quarter. Revenue grew year-over-year, 
# driven by robust demand across our product portfolio. Operating margins expanded as we 
# maintained disciplined cost management while investing in growth initiatives.

# Looking at the numbers, total revenue reached a new record, with particularly strong 
# performance in our core segments. We continued to generate significant free cash flow, 
# which we deployed toward shareholder returns and strategic investments.

# Despite macroeconomic headwinds, our business demonstrated resilience. We're seeing 
# strong customer engagement and positive trends in key metrics. Our innovation pipeline 
# remains robust, and we're well-positioned for sustainable growth.

# For the full year, we're maintaining our guidance and remain confident in our ability 
# to deliver value for shareholders. We see significant opportunities ahead and are 
# executing against our strategic priorities.""",
            
#             f"""Thank you for joining us today to discuss {ticker}'s {quarter} results.

# Our performance this quarter reflects the strength of our business model and the 
# dedication of our team. We delivered solid results across key financial metrics, 
# demonstrating operational excellence and disciplined execution.

# Revenue growth was balanced across geographies and business segments. We saw particular 
# strength in our high-margin product categories. Operating efficiency improved, resulting 
# in year-over-year margin expansion. We generated strong cash flow, enabling continued 
# investment in innovation while returning capital to shareholders.

# The competitive landscape remains dynamic, but we're confident in our differentiated 
# value proposition. Customer satisfaction metrics are trending positively, and we're 
# gaining market share in key categories. Our strategic initiatives are on track, and 
# we're making good progress against our long-term objectives.

# Looking ahead, we see both opportunities and challenges. We're focused on execution, 
# innovation, and maintaining financial discipline."""
#         ]
        
#         return templates[hash(ticker) % len(templates)]
    
#     def _generate_sample_qa(self, ticker, quarter):
#         """Generate realistic sample Q&A section"""
        
#         qa_template = f"""
# QUESTION AND ANSWER SESSION

# Analyst 1: Thank you for taking my question. Can you provide more color on the revenue 
# growth drivers this quarter? What are you seeing in terms of demand trends?

# Management: Thanks for the question. The revenue growth was broad-based. We saw strength 
# across multiple product lines and geographies. Customer demand remains healthy, though 
# we're monitoring the macro environment closely. Our sales pipeline is robust, and we're 
# seeing good conversion rates.

# Analyst 1: And on margins, how sustainable is the current level given input cost pressures?

# Management: We're focused on operational efficiency and productivity. While we face some 
# cost headwinds, our pricing strategies and operational improvements are helping us maintain 
# margins. We expect to sustain current levels through disciplined cost management.

# Analyst 2: Can you talk about your capital allocation priorities and what shareholders 
# should expect regarding buybacks and dividends?

# Management: Capital allocation remains a priority. We generated strong free cash flow 
# this quarter, which gives us flexibility. We're committed to our dividend and will 
# continue opportunistic share repurchases while maintaining investment in growth 
# initiatives. We evaluate opportunities based on return profiles and strategic fit.

# Analyst 2: Thank you. And one more on competition - are you seeing any changes in the 
# competitive dynamics?

# Management: The competitive environment is always evolving, but our value proposition 
# remains strong. We're focused on innovation, customer service, and operational excellence. 
# Our market position is solid, and we're confident in our ability to compete effectively.

# Thank you all for joining today's call. We appreciate your continued interest in {ticker}.
# """
        
#         return qa_template
    
#     # def _get_quarter_date(self, quarter_str):
#     #     """Convert quarter string to date that matches financial data"""
#     #     quarter_map = {
#     #         'Q2 2024': '2024-07-31',
#     #         'Q3 2024': '2024-10-31', 
#     #         'Q4 2024': '2025-01-31',
#     #     }
#     #     return quarter_map.get(quarter_str, '2024-10-31')
#     def _get_quarter_date(self, quarter_str):
#         quarter, year = quarter_str.split()
#         year = int(year)
#         quarter_num = int(quarter[1])

#         quarter_end = {
#             1: f"{year}-03-31",
#             2: f"{year}-06-30",
#             3: f"{year}-09-30",
#             4: f"{year}-12-31"
#         }

#         return quarter_end[quarter_num]
    
#     def save_transcripts(self, transcripts):
#         """
#         Save transcripts to both JSON and CSV formats
#         """
#         if not transcripts:
#             print("❌ No transcripts to save")
#             return False
        
#         try:
#             print("="*60)
#             print("💾 SAVING TRANSCRIPTS")
#             print("="*60)
#             print()
            
#             # Convert to DataFrame
#             df = pd.DataFrame(transcripts)
            
#             # Save as CSV (for easy viewing)
#             csv_path = self.output_dir / 'transcripts_metadata.csv'
#             metadata_cols = ['ticker', 'company_name', 'quarter', 'date', 'fiscal_year', 
#                            'fiscal_quarter', 'word_count', 'source', 'collection_date']
#             df[metadata_cols].to_csv(csv_path, index=False)
#             print(f"✅ Metadata saved: {csv_path}")
            
#             # Save full transcripts as JSON (preserves structure)
#             json_path = self.output_dir / 'transcripts_full.json'
#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(transcripts, f, indent=2, ensure_ascii=False)
#             print(f"✅ Full transcripts saved: {json_path}")
#             print(f"   Size: {json_path.stat().st_size / (1024*1024):.2f} MB")
            
#             # Save individual transcript files for easy access
#             individual_dir = self.output_dir / 'individual'
#             individual_dir.mkdir(exist_ok=True)
            
#             for transcript in transcripts:
#                 filename = f"{transcript['ticker']}_{transcript['quarter'].replace(' ', '_')}.txt"
#                 file_path = individual_dir / filename
                
#                 with open(file_path, 'w', encoding='utf-8') as f:
#                     f.write(f"Company: {transcript['company_name']}\n")
#                     f.write(f"Ticker: {transcript['ticker']}\n")
#                     f.write(f"Quarter: {transcript['quarter']}\n")
#                     f.write(f"Date: {transcript['date']}\n")
#                     f.write("="*60 + "\n\n")
#                     f.write(transcript['full_text'])
            
#             print(f"✅ Individual transcripts saved: {individual_dir}/")
#             print(f"   Count: {len(transcripts)} files")
            
#             print()
#             return True
            
#         except Exception as e:
#             print(f"❌ Error saving transcripts: {e}")
#             return False
    
#     def _log_info(self, message):
#         """Log info"""
#         with open(self.log_file, 'a') as f:
#             f.write(f"{datetime.now()} - INFO: {message}\n")
    
#     def _log_error(self, message):
#         """Log errors"""
#         with open(self.log_file, 'a') as f:
#             f.write(f"{datetime.now()} - ERROR: {message}\n")
    
#     def run(self):
#         """
#         Execute complete transcript collection workflow
#         """
#         print("="*60)
#         print("PHASE 1D: EARNINGS CALL TRANSCRIPT COLLECTION")
#         print("="*60)
#         print()
#         print(f"🎯 Target: {self.target_count} transcripts")
#         print()
        
#         # Get priority tickers
#         tickers = self.get_high_priority_tickers()
#         print(f"📊 Priority tickers: {len(tickers)}")
#         print()
        
#         # Strategy explanation
#         print("📋 COLLECTION STRATEGY:")
#         print("   1. Sample transcripts for pipeline development")
#         print("   2. Manual collection from public sources (recommended)")
#         print("   3. Scalable design for future expansion")
#         print()
        
#         # Create sample transcripts
#         transcripts = self.create_sample_transcripts(tickers, transcripts_per_ticker=3)
        
#         # Save transcripts
#         if transcripts:
#             success = self.save_transcripts(transcripts)
            
#             if success:
#                 print("="*60)
#                 print("✅ PHASE 1D COMPLETE")
#                 print("="*60)
#                 print()
#                 print(f"✅ Collected: {len(transcripts)} transcripts")
#                 print(f"✅ Unique companies: {len(set(t['ticker'] for t in transcripts))}")
#                 print(f"✅ Average word count: {sum(t['word_count'] for t in transcripts) / len(transcripts):.0f}")
#                 print()
#                 print("📁 Data saved to:", self.output_dir)
#                 print()
#                 print("="*60)
#                 print("⚠️  IMPORTANT: NEXT STEPS FOR REAL TRANSCRIPTS")
#                 print("="*60)
#                 print()
#                 print("These are SAMPLE transcripts for pipeline development.")
#                 print()
#                 print("To add REAL transcripts:")
#                 print("1. Visit: https://seekingalpha.com/earnings/earnings-call-transcripts")
#                 print("2. Manually download 20-30 recent transcripts (free tier)")
#                 print("3. Save as .txt files in: data/raw/transcripts/individual/")
#                 print("4. Run the parser to process them")
#                 print()
#                 print("OR use provided manual collection script (next phase)")
#                 print()
                
#                 return transcripts
        
#         return None

# def main():
#     """Main execution"""
#     collector = TranscriptCollector(target_count=150)
#     transcripts = collector.run()
#     return transcripts

# if __name__ == "__main__":
#     main()

"""
Phase 1D: Earnings Call Transcript Collection
Collects earnings call transcripts from multiple sources
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import time
import json
import re
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, LOGS_DIR

class TranscriptCollector:
    """
    Collects earnings call transcripts from multiple sources
    """
    
    def __init__(self, target_count=150):
        self.target_count = target_count
        
        self.output_dir = RAW_DATA_DIR / 'transcripts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'data_collection' / 'transcripts.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.collected_transcripts = []
        self.failed_attempts = []
        
        # Headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def get_high_priority_tickers(self):
        """
        Get tickers that ACTUALLY have financial data
        This ensures alignment in Phase 2D
        """
        from config import PROCESSED_DATA_DIR
        import pandas as pd
        
        # Read tickers from actual financial data
        financial_file = PROCESSED_DATA_DIR / 'financial' / 'financial_features_normalized.csv'
        
        if financial_file.exists():
            print("📊 Reading tickers from financial data...")
            df = pd.read_csv(financial_file)
            available_tickers = df['Ticker'].unique().tolist()
            print(f"✅ Found {len(available_tickers)} tickers with financial data")
            
            # Return first 50 tickers that have the most data
            ticker_counts = df['Ticker'].value_counts()
            top_tickers = ticker_counts.head(50).index.tolist()
            
            return top_tickers
        else:
            print("⚠️  Financial data not found, using default tickers")
            # Fallback to default list
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
                'XOM', 'V', 'JPM', 'PG', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV',
                'PEP', 'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
                'BAC', 'CRM', 'ADBE', 'DHR', 'LIN', 'NFLX', 'VZ', 'NKE', 'TXN', 'CMCSA',
                'WFC', 'NEE', 'DIS', 'RTX', 'ORCL', 'AMD', 'UPS', 'PM', 'QCOM', 'MS'
            ]
    
    def collect_from_sec_edgar(self, ticker, max_transcripts=3):
        """
        Collect transcripts from SEC EDGAR (8-K filings)
        These often contain earnings call information
        """
        collected = []
        
        try:
            # SEC EDGAR search URL
            # Note: This is a simplified version. Real implementation would use SEC API
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=exclude&count=10"
            
            print(f"   Attempting SEC EDGAR for {ticker}...")
            
            # For now, we'll mark this as a placeholder
            # Real implementation would parse SEC filings
            self._log_info(f"SEC EDGAR: {ticker} - Placeholder (requires SEC API access)")
            
            return collected
            
        except Exception as e:
            self._log_error(f"SEC EDGAR {ticker}: {e}")
            return collected
    
    def collect_from_seeking_alpha_sample(self, ticker, max_transcripts=3):
        """
        Attempt to collect transcripts from Seeking Alpha
        CAUTIOUS approach with rate limiting
        """
        collected = []
        
        try:
            # Seeking Alpha earnings URL pattern
            base_url = "https://seekingalpha.com/symbol"
            url = f"{base_url}/{ticker}/earnings/transcripts"
            
            print(f"   Attempting Seeking Alpha for {ticker}...")
            
            # Make request with headers
            response = requests.get(url, headers=self.headers, timeout=10)
            
            # Check if blocked
            if response.status_code == 403:
                print(f"   ⚠️  Blocked by Seeking Alpha for {ticker}")
                self._log_error(f"Seeking Alpha {ticker}: 403 Forbidden")
                return collected
            
            if response.status_code != 200:
                print(f"   ⚠️  Failed to access {ticker}: Status {response.status_code}")
                return collected
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is simplified - actual parsing would need to match SA's HTML structure
            # For demonstration, we'll create placeholder structure
            print(f"   ⚠️  Seeking Alpha parsing requires authentication - using placeholder")
            self._log_info(f"Seeking Alpha {ticker}: Requires authentication")
            
            return collected
            
        except Exception as e:
            self._log_error(f"Seeking Alpha {ticker}: {e}")
            return collected
    
    def create_sample_transcripts(self, tickers, transcripts_per_ticker=3):
        """
        Create sample/demo transcripts for proof-of-concept
        This allows the project to proceed while real transcripts are collected manually
        """
        print("="*60)
        print("CREATING SAMPLE TRANSCRIPTS")
        print("="*60)
        print()
        print("⚠️  Note: These are structured samples for pipeline development")
        print("   Real transcripts should be added manually to data/raw/transcripts/")
        print()
        
        sample_transcripts = []
        
        quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
        
        for ticker in tqdm(tickers[:50], desc="Generating sample transcripts"):  # 50 tickers x 3 = 150
            for i in range(min(transcripts_per_ticker, len(quarters))):
                quarter = quarters[i]
                
                # Create realistic sample transcript structure
                transcript = {
                    'ticker': ticker,
                    'company_name': f"{ticker} Inc.",
                    'quarter': quarter,
                    'date': self._get_quarter_date(quarter),
                    'fiscal_year': 2023,
                    'fiscal_quarter': i + 1,
                    
                    # Prepared Remarks (Management)
                    'prepared_remarks': self._generate_sample_prepared_remarks(ticker, quarter),
                    
                    # Q&A Section
                    'qa_section': self._generate_sample_qa(ticker, quarter),
                    
                    # Full transcript
                    'full_text': '',
                    
                    # Metadata
                    'word_count': 0,
                    'speaker_count': 0,
                    'source': 'SAMPLE',
                    'collection_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                # Combine sections
                transcript['full_text'] = transcript['prepared_remarks'] + "\n\n" + transcript['qa_section']
                transcript['word_count'] = len(transcript['full_text'].split())
                
                sample_transcripts.append(transcript)
        
        print(f"\n✅ Generated {len(sample_transcripts)} sample transcripts")
        print()
        
        return sample_transcripts
    
    def _generate_sample_prepared_remarks(self, ticker, quarter):
        """Generate varied prepared remarks with different tones and sentiments"""
        import random
        rng = random.Random(hash(ticker + quarter))

        # Sector-specific context
        sector_context = {
            'tech': ['cloud computing adoption', 'AI integration', 'software subscriptions', 'developer ecosystem'],
            'finance': ['net interest margin', 'loan growth', 'credit quality', 'fee income'],
            'health': ['drug pipeline', 'clinical trials', 'regulatory approvals', 'patient outcomes'],
            'consumer': ['same-store sales', 'foot traffic', 'digital commerce', 'brand loyalty'],
            'energy': ['production volumes', 'refining margins', 'commodity prices', 'capital discipline'],
            'industrial': ['backlog growth', 'pricing realization', 'supply chain', 'automation demand'],
        }
        sector_keys = list(sector_context.keys())
        sector = sector_keys[hash(ticker) % len(sector_keys)]
        metrics = sector_context[sector]
        m1, m2 = rng.sample(metrics, 2)

        # Revenue numbers
        rev = rng.choice(['$4.2 billion', '$7.8 billion', '$2.1 billion', '$12.4 billion', '$890 million', '$3.6 billion'])
        rev_growth = rng.choice(['+3%', '+8%', '+14%', '+2%', '-1%', '+22%', '+6%', '-4%'])
        margin = rng.choice(['18.2%', '24.7%', '31.1%', '12.4%', '8.9%', '42.3%', '19.8%'])
        eps = rng.choice(['$1.24', '$3.87', '$0.94', '$2.15', '$5.32', '$0.67', '$1.89'])

        # Choose sentiment scenario
        scenario = rng.randint(0, 5)

        if scenario == 0:  # Strong beat, very positive
            tone = f"""Good morning, and thank you for joining {ticker}'s {quarter} earnings call.

I am thrilled to report an exceptional quarter that significantly exceeded our expectations. 
Revenue came in at {rev}, representing {rev_growth} year-over-year growth, beating consensus 
estimates by a wide margin. Our {m1} business delivered record performance, and {m2} 
continues to be a meaningful growth driver.

Operating margins expanded to {margin}, a 280 basis point improvement versus the prior year. 
Diluted EPS of {eps} reflects the operational leverage in our business model. Free cash flow 
generation was outstanding, and we returned significant capital to shareholders through 
buybacks and dividends.

Customer demand is accelerating. We are seeing strong adoption across all geographies, 
and our pipeline has never been stronger. We are raising full-year guidance and are 
highly confident in our ability to deliver sustained, profitable growth.

This is an exciting time for {ticker}. Our competitive position has never been stronger, 
and the tailwinds in our end markets are building. We are investing aggressively in 
innovation while maintaining financial discipline."""

        elif scenario == 1:  # Miss, negative tone
            tone = f"""Good morning. Thank you for joining {ticker}'s {quarter} earnings call.

I want to be direct with you today. Our results for the quarter came in below our 
expectations and below what you deserve from us. Revenue of {rev} was {rev_growth} 
year-over-year, which fell short of both our internal targets and consensus estimates.

We faced a challenging environment. {m1.capitalize()} deteriorated more rapidly than 
anticipated, and the headwinds in {m2} proved more persistent than we modeled. Margins 
compressed to {margin} due to elevated costs and pricing pressure that we were unable 
to fully offset.

EPS of {eps} reflects these operational challenges. We are not satisfied with this 
performance and are taking immediate action. We have initiated a cost reduction program, 
are rationalizing our portfolio, and are making leadership changes in underperforming 
business units.

We are lowering our full-year guidance to reflect current market realities. I want to 
assure shareholders that we are clear-eyed about our challenges and are taking the 
necessary steps to restore performance. This is a difficult quarter, but we believe 
the long-term fundamentals of our business remain intact."""

        elif scenario == 2:  # Mixed results, cautious tone
            tone = f"""Thank you for joining us today to discuss {ticker}'s {quarter} results.

Our performance this quarter was mixed. Revenue of {rev} grew {rev_growth} year-over-year, 
in line with expectations, though we saw meaningful divergence across our business segments. 
Our {m1} business performed well, while {m2} faced headwinds that we are actively managing.

Operating margins of {margin} were roughly flat sequentially. We are navigating a complex 
macro environment and are being thoughtful about investments. EPS of {eps} reflects 
disciplined cost management in areas within our control, partially offsetting top-line 
softness in certain geographies.

We are maintaining our full-year guidance range, though we acknowledge the range of 
outcomes has widened. Consumer sentiment is uncertain, and we are scenario planning 
for multiple environments. Our balance sheet is strong, which gives us flexibility.

Looking ahead, we see a mix of opportunities and risks. We remain focused on execution, 
cost discipline, and protecting our competitive position while preserving the ability 
to invest when conditions improve."""

        elif scenario == 3:  # Turnaround story, cautiously optimistic
            tone = f"""Good morning, and thank you for joining {ticker}'s {quarter} earnings call.

I am pleased to report that our turnaround is gaining momentum. Revenue of {rev} 
represents {rev_growth} growth — the third consecutive quarter of sequential improvement. 
After a difficult period, the strategic actions we took are beginning to yield results.

Our restructuring is substantially complete, and we are now operating with a leaner, 
more focused cost structure. Margins recovered to {margin}, up significantly from the 
trough levels we experienced. EPS of {eps} demonstrates that profitability is returning.

The recovery in {m1} has been particularly encouraging. While {m2} remains below 
historical levels, the rate of decline has moderated substantially. We are cautiously 
optimistic that we have seen the bottom in these markets.

We are reinstating guidance for the full year and see a credible path to returning to 
our long-term financial targets. There is still work to do, but I am more confident 
today than I have been in several quarters that {ticker} is on the right trajectory."""

        elif scenario == 4:  # High growth, investment mode
            tone = f"""Thank you for joining {ticker}'s {quarter} earnings call.

We delivered another strong quarter of top-line growth with revenue of {rev}, up {rev_growth} 
year-over-year. Our {m1} segment is scaling exceptionally well, and {m2} is becoming 
an increasingly meaningful contributor to our overall business.

We are intentionally investing ahead of the growth curve, which is why margins of {margin} 
reflect higher operating expenses. We believe the long-term returns on these investments 
will be substantial, and we are prioritizing growth over near-term profitability. 
EPS of {eps} reflects this deliberate investment posture.

Customer acquisition and retention metrics are at all-time highs. Net revenue retention 
exceeds 120%, and our logo count grew by over 30% year-over-year. The addressable market 
opportunity is massive, and we are capturing share at an accelerating pace.

We are raising our revenue growth outlook for the full year. We see no slowdown in 
demand, and our competitive moat is widening. This is the right moment to invest 
aggressively, and we are doing exactly that."""

        else:  # Macro uncertainty, defensive
            tone = f"""Good morning. Thank you for joining us for {ticker}'s {quarter} earnings call.

We operated in a challenging macro environment this quarter and I believe we navigated 
it reasonably well. Revenue of {rev} was {rev_growth} versus the prior year period. 
While growth moderated from recent quarters, we maintained profitability and cash generation.

We are seeing a more cautious purchasing environment, particularly in {m1}. Enterprise 
customers are extending decision cycles and scrutinizing budgets more carefully. 
{m2.capitalize()} showed more resilience, which partially offset softness elsewhere.

Margins held at {margin} through tight cost discipline. We reduced discretionary spending, 
slowed hiring, and deferred certain capital projects. EPS of {eps} reflects these 
efficiency efforts. Free cash flow remained solid, and our balance sheet is a source 
of strength in this environment.

We are widening our guidance range to reflect macro uncertainty. We are not in a position 
to predict when conditions will improve, but we are managing the business conservatively 
and are well-positioned to emerge stronger when the environment stabilizes."""

        return tone
    
    def _generate_sample_qa(self, ticker, quarter):
        """Generate varied Q&A sections with different analyst tones"""
        import random
        rng = random.Random(hash(ticker + quarter + "qa"))
        scenario = rng.randint(0, 4)

        if scenario == 0:  # Aggressive analyst questioning, defensive management
            qa = f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Goldman Sachs): I appreciate the prepared remarks, but I want to push back 
a little. The revenue miss was meaningful and guidance implies continued deceleration. 
Can you help us understand what specifically went wrong and why we should have confidence 
the back half improves?

Management: I understand the skepticism and I think it's warranted. We did miss, and 
I'm not going to sugarcoat that. The primary driver was elongated sales cycles in our 
enterprise segment — deals that we expected to close slipped into next quarter. We have 
line of sight to those deals. Underlying demand has not deteriorated; it's a timing issue.

Analyst 1: And if those deals slip again?

Management: That's a fair challenge. We've restructured our sales compensation to 
penalize slippage and are requiring more rigorous stage-gate discipline in our CRM. 
We're not banking on hope — we've made structural changes.

Analyst 2 (Morgan Stanley): On margins — you talked about cost discipline but SG&A 
actually went up as a percentage of revenue. Can you walk us through that?

Management: You're right to flag it. We had some one-time items in SG&A — severance 
costs from the reorganization and some legal accruals. Stripping those out, core SG&A 
was down year-over-year. We should see a cleaner picture in Q4.

Analyst 2: What's the right normalized margin target for {ticker} in this environment?

Management: We're targeting a return to the mid-twenties operating margin within 
four to six quarters. We'll provide more specificity at our investor day in the spring.

Thank you all. We appreciate your tough but fair questions today."""

        elif scenario == 1:  # Enthusiastic analysts, confident management
            qa = f"""
QUESTION AND ANSWER SESSION

Analyst 1 (JPMorgan): Fantastic quarter. The beat was broad-based and the raise was 
larger than we expected. Can you talk about what's driving the upside surprise?

Management: Thank you. Three things really came together this quarter. First, our 
go-to-market changes from eighteen months ago are now in full effect — we're seeing 
much stronger attach rates on our premium offerings. Second, international markets 
accelerated meaningfully, particularly in EMEA. Third, gross margins benefited from 
a favorable mix shift that we expect to persist.

Analyst 1: Is the guidance raise conservative? It seems like you have momentum.

Management: We think it's appropriately achievable. We don't like to over-promise. 
There are still macro uncertainties and we want to leave room to outperform rather 
than setting an expectation we might miss.

Analyst 2 (Bank of America): On capital allocation — you've now built up a substantial 
cash position. What's the priority use?

Management: Organic investment remains priority one. We have a rich internal pipeline 
and the returns on organic R&D are excellent. Beyond that, we'll consider tuck-in 
acquisitions in adjacent markets, and we'll continue returning capital through buybacks. 
We're very disciplined on M&A — we've passed on several deals that didn't meet our 
return thresholds.

Analyst 2: Any specific geographies or product areas you're most excited about heading 
into next year?

Management: Asia Pacific is an underappreciated opportunity for us. We've been investing 
in go-to-market infrastructure there for two years and I think you'll see that begin 
to show up in results. On product, our next-generation platform launch is on track and 
customer feedback from beta has been exceptional.

Thank you all for joining. Exciting times at {ticker}."""

        elif scenario == 2:  # Cautious analysts, measured management
            qa = f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Citi): Thank you for taking my question. The results were largely in line 
but I'm trying to understand the sustainability of margins at this level given the 
cost environment. What gives you confidence?

Management: The margin performance reflects a few durable factors. We've permanently 
taken out structural costs through our efficiency program — that's roughly 150 basis 
points of permanent savings. And our pricing actions from earlier this year are now 
fully in the run rate. Those are sustainable. What's less predictable is input cost 
inflation, which we're actively hedging.

Analyst 1: And if volumes disappoint in the second half, do you have incremental 
levers to protect earnings?

Management: Yes. We have a variable cost structure that allows us to flex down relatively 
quickly. We also have a discretionary spending reserve that we've identified but not 
yet cut — that's an available lever. We're not planning to use it, but it's there.

Analyst 2 (Wells Fargo): Can you give us an update on the competitive environment? 
There's been some aggressive pricing from a key competitor.

Management: We're aware of the pricing moves. Our win rates have remained stable, which 
tells us customers value our differentiation. We're not going to race to the bottom 
on price — that's not a game we need to play. Our value proposition is strong enough 
to command a premium, and we intend to defend that.

Analyst 2: Last one — any update on the CEO search?

Management: The search is progressing well. The board is engaged and we expect to have 
an announcement in the coming months. In the interim, the business is running well 
and the team is executing.

Thank you everyone for your time today."""

        elif scenario == 3:  # Concern about guidance, detailed questioning
            qa = f"""
QUESTION AND ANSWER SESSION

Analyst 1 (UBS): The guidance midpoint implies a meaningful step-down in growth. 
Can you help us bridge from current trends to that implied deceleration?

Management: Sure. A few factors. First, we have a tougher year-over-year comparison 
in the back half — we had a particularly strong Q3 last year from a large one-time 
government contract that won't repeat. Second, we're being more conservative on 
the macro given the interest rate environment. And third, we're choosing to invest 
more heavily in the business in Q4, which will weigh on near-term earnings.

Analyst 1: On the government contract point — is that fully reflected in consensus?

Management: I'm not sure what's in every model, but we've been consistent in flagging 
that comparison. It was disclosed in last year's K. That quarter was roughly $180 million 
of revenue above our typical run rate.

Analyst 2 (Deutsche Bank): On the balance sheet — debt levels are elevated relative 
to your historical comfort zone. What's the path to deleveraging?

Management: We're committed to getting back to our target leverage range of 2 to 2.5 
times within 18 months. Free cash flow is our primary tool — we're prioritizing debt 
paydown over buybacks for the next several quarters. We may also consider non-core 
asset divestitures if we can achieve attractive valuations.

Analyst 2: What assets are you considering divesting?

Management: We don't want to comment specifically as we're in early stages of analysis. 
But we're looking at businesses where we are not the natural owner and where a strategic 
buyer would likely ascribe higher value.

Thank you for joining us today for {ticker}'s earnings call."""

        else:  # Short, terse Q&A with skeptical tone
            qa = f"""
QUESTION AND ANSWER SESSION

Analyst 1 (Barclays): I'll be brief. Three questions. One: what's the right revenue 
run rate exiting the year? Two: what's your realistic floor on margins? Three: when 
does the free cash flow conversion normalize?

Management: On revenue exit rate — we expect sequential improvement each quarter 
through year-end, implying a Q4 exit rate in the $1.1 to $1.2 billion range annualized. 
On margin floor — we believe we've seen it; the cost actions we've taken create a floor 
in the high single digits. On free cash flow — working capital normalization should 
drive conversion back above 90% by Q4.

Analyst 1: Are you confident in those numbers or is that aspirational?

Management: I'm confident. Those are numbers I'm prepared to be held accountable to.

Analyst 2 (Bernstein): The stock is down significantly year-to-date. Does management 
feel any urgency around capital return given the valuation?

Management: We're acutely aware of the share price. The board has authorized an 
incremental $500 million buyback, and we intend to be active in the open market. 
We think the stock is meaningfully undervalued at current levels, and we're putting 
our capital where our mouth is.

Analyst 2: Thank you. I'll leave it there.

Thank you all. We appreciate the engagement and look forward to updating you on our 
progress next quarter at {ticker}."""

        return qa
    
    # def _get_quarter_date(self, quarter_str):
    #     """Convert quarter string to date that matches financial data"""
    #     quarter_map = {
    #         'Q2 2024': '2024-07-31',
    #         'Q3 2024': '2024-10-31', 
    #         'Q4 2024': '2025-01-31',
    #     }
    #     return quarter_map.get(quarter_str, '2024-10-31')
    def _get_quarter_date(self, quarter_str):
        quarter, year = quarter_str.split()
        year = int(year)
        quarter_num = int(quarter[1])

        quarter_end = {
            1: f"{year}-03-31",
            2: f"{year}-06-30",
            3: f"{year}-09-30",
            4: f"{year}-12-31"
        }

        return quarter_end[quarter_num]
    
    def save_transcripts(self, transcripts):
        """
        Save transcripts to both JSON and CSV formats
        """
        if not transcripts:
            print("❌ No transcripts to save")
            return False
        
        try:
            print("="*60)
            print("💾 SAVING TRANSCRIPTS")
            print("="*60)
            print()
            
            # Convert to DataFrame
            df = pd.DataFrame(transcripts)
            
            # Save as CSV (for easy viewing)
            csv_path = self.output_dir / 'transcripts_metadata.csv'
            metadata_cols = ['ticker', 'company_name', 'quarter', 'date', 'fiscal_year', 
                           'fiscal_quarter', 'word_count', 'source', 'collection_date']
            df[metadata_cols].to_csv(csv_path, index=False)
            print(f"✅ Metadata saved: {csv_path}")
            
            # Save full transcripts as JSON (preserves structure)
            json_path = self.output_dir / 'transcripts_full.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, indent=2, ensure_ascii=False)
            print(f"✅ Full transcripts saved: {json_path}")
            print(f"   Size: {json_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save individual transcript files for easy access
            individual_dir = self.output_dir / 'individual'
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
            
            print(f"✅ Individual transcripts saved: {individual_dir}/")
            print(f"   Count: {len(transcripts)} files")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Error saving transcripts: {e}")
            return False
    
    def _log_info(self, message):
        """Log info"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - INFO: {message}\n")
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete transcript collection workflow
        """
        print("="*60)
        print("PHASE 1D: EARNINGS CALL TRANSCRIPT COLLECTION")
        print("="*60)
        print()
        print(f"🎯 Target: {self.target_count} transcripts")
        print()
        
        # Get priority tickers
        tickers = self.get_high_priority_tickers()
        print(f"📊 Priority tickers: {len(tickers)}")
        print()
        
        # Strategy explanation
        print("📋 COLLECTION STRATEGY:")
        print("   1. Sample transcripts for pipeline development")
        print("   2. Manual collection from public sources (recommended)")
        print("   3. Scalable design for future expansion")
        print()
        
        # Create sample transcripts
        transcripts = self.create_sample_transcripts(tickers, transcripts_per_ticker=3)
        
        # Save transcripts
        if transcripts:
            success = self.save_transcripts(transcripts)
            
            if success:
                print("="*60)
                print("✅ PHASE 1D COMPLETE")
                print("="*60)
                print()
                print(f"✅ Collected: {len(transcripts)} transcripts")
                print(f"✅ Unique companies: {len(set(t['ticker'] for t in transcripts))}")
                print(f"✅ Average word count: {sum(t['word_count'] for t in transcripts) / len(transcripts):.0f}")
                print()
                print("📁 Data saved to:", self.output_dir)
                print()
                print("="*60)
                print("⚠️  IMPORTANT: NEXT STEPS FOR REAL TRANSCRIPTS")
                print("="*60)
                print()
                print("These are SAMPLE transcripts for pipeline development.")
                print()
                print("To add REAL transcripts:")
                print("1. Visit: https://seekingalpha.com/earnings/earnings-call-transcripts")
                print("2. Manually download 20-30 recent transcripts (free tier)")
                print("3. Save as .txt files in: data/raw/transcripts/individual/")
                print("4. Run the parser to process them")
                print()
                print("OR use provided manual collection script (next phase)")
                print()
                
                return transcripts
        
        return None

def main():
    """Main execution"""
    collector = TranscriptCollector(target_count=150)
    transcripts = collector.run()
    return transcripts

if __name__ == "__main__":
    main()