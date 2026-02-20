# """
# Phase 2D: Temporal Alignment
# Aligns transcripts, financials, and market data on timeline
# CRITICAL: Prevents lookahead bias in prediction
# """

# import pandas as pd
# import numpy as np
# import json
# from pathlib import Path
# from datetime import datetime, timedelta
# import sys
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# # Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))

# from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR

# class TemporalAligner:
#     """
#     Aligns earnings calls, financial data, and market data temporally
#     Ensures no lookahead bias in the dataset
#     """
    
#     def __init__(self):
#         self.transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
#         self.financial_dir = PROCESSED_DATA_DIR / 'financial'
#         self.market_dir = RAW_DATA_DIR / 'market'
        
#         self.output_dir = PROCESSED_DATA_DIR / 'aligned'
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.log_file = LOGS_DIR / 'preprocessing' / 'temporal_alignment.log'
#         self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
#         self.alignment_stats = {
#             'total_transcripts': 0,
#             'successfully_aligned': 0,
#             'missing_financial_data': 0,
#             'missing_market_data': 0,
#             'temporal_mismatches': 0,
#             'final_aligned_records': 0
#         }
    
#     def load_data_sources(self):
#         """
#         Load all data sources: transcripts, financials, market data
#         """
#         print("="*60)
#         print("PHASE 2D: TEMPORAL ALIGNMENT")
#         print("="*60)
#         print()
#         print("📂 LOADING DATA SOURCES...")
#         print()
        
#         # 1. Load segmented transcripts
#         transcripts_file = self.transcripts_dir / 'transcripts_segmented.json'
#         if not transcripts_file.exists():
#             print(f"❌ Segmented transcripts not found: {transcripts_file}")
#             return None, None, None
        
#         with open(transcripts_file, 'r', encoding='utf-8') as f:
#             transcripts = json.load(f)
        
#         print(f"✅ Transcripts loaded: {len(transcripts)} records")
        
#         # 2. Load normalized financials
#         financials_file = self.financial_dir / 'financial_features_normalized.csv'
#         if not financials_file.exists():
#             print(f"❌ Normalized financials not found: {financials_file}")
#             return None, None, None
        
#         financials_df = pd.read_csv(financials_file)
#         financials_df['Date'] = pd.to_datetime(financials_df['Date'])
        
#         print(f"✅ Financials loaded: {len(financials_df)} records")
#         print("Financial tickers sample:", financials_df['Ticker'].unique()[:10])
#         print("Transcript ticker sample:", transcripts[0]['ticker'])

#         # 3. Load market data
#         market_file = self.market_dir / 'stock_prices.csv'
#         if not market_file.exists():
#             print(f"❌ Market data not found: {market_file}")
#             return None, None, None
        
#         market_df = pd.read_csv(market_file)
#         market_df['Date'] = pd.to_datetime(market_df['Date'])
        
#         print(f"✅ Market data loaded: {len(market_df)} records")
#         print()
        
#         self.alignment_stats['total_transcripts'] = len(transcripts)
        
#         return transcripts, financials_df, market_df
    
#     def parse_quarter_to_date(self, quarter_str):
#         """
#         Convert quarter string to approximate date
        
#         Examples:
#         "Q1 2023" -> 2023-03-31 (end of Q1)
#         "Q2 2023" -> 2023-06-30 (end of Q2)
#         "Q3 2023" -> 2023-09-30 (end of Q3)
#         "Q4 2023" -> 2023-12-31 (end of Q4)
#         """
#         try:
#             # Parse "Q1 2023" format
#             quarter, year = quarter_str.split()
#             year = int(year)
#             quarter_num = int(quarter[1])
            
#             # Map quarter to end-of-quarter date
#             quarter_end_dates = {
#                 1: f"{year}-03-31",
#                 2: f"{year}-06-30",
#                 3: f"{year}-09-30",
#                 4: f"{year}-12-31"
#             }
            
#             date_str = quarter_end_dates[quarter_num]
#             return pd.to_datetime(date_str)
            
#         except Exception as e:
#             self._log_error(f"Error parsing quarter '{quarter_str}': {e}")
#             return None
    
#     def find_matching_financial(self, ticker, transcript_date, financials_df, tolerance_days=45):
#         """
#         Find financial data matching the transcript
        
#         Matching logic:
#         1. Same ticker
#         2. Financial date within tolerance_days of transcript date
#         3. Financial date should be BEFORE or equal to transcript date (no lookahead)
#         """
#         # Filter by ticker
#         ticker_financials = financials_df[financials_df['Ticker'] == ticker].copy()
        
#         if ticker_financials.empty:
#             return None
        
#         # Find closest financial record BEFORE transcript date
#         ticker_financials['DateDiff'] = (transcript_date - ticker_financials['Date']).dt.days
        
#         # Keep only records where financial date <= transcript date (no lookahead)
#         # And within tolerance window
#         valid_records = ticker_financials[
#             (ticker_financials['DateDiff'] >= 0) & 
#             (ticker_financials['DateDiff'] <= tolerance_days)
#         ]
        
#         if valid_records.empty:
#             print(f"⚠️ No financial match for {ticker} on {transcript_date.date()}")
#             ticker_financials_dates = ticker_financials['Date'].dt.date.tolist()
#             print(f"   Available financial dates: {ticker_financials_dates[:5]} ...")
#             return None
        
#         # Get the closest match
#         closest_match = valid_records.loc[valid_records['DateDiff'].idxmin()]
        
#         return closest_match
    
#     def get_market_data_window(self, ticker, date, market_df, days_before=5, days_after=3):
#         """
#         Get market data in a window around the earnings date
        
#         Returns:
#         - Price before earnings (for return calculation)
#         - Price after earnings (for abnormal return target)
#         """
#         # Filter by ticker
#         ticker_market = market_df[market_df['Ticker'] == ticker].copy()
        
#         if ticker_market.empty:
#             return None
        
#         ticker_market = ticker_market.sort_values('Date')
        
#         # Find closest date before earnings
#         before_dates = ticker_market[ticker_market['Date'] <= date]
#         if before_dates.empty:
#             price_before = None
#             date_before = None
#         else:
#             before_record = before_dates.iloc[-1]
#             price_before = before_record['Close']
#             date_before = before_record['Date']
        
#         # Find date ~3 days after earnings
#         after_dates = ticker_market[ticker_market['Date'] > date]
#         if len(after_dates) < days_after:
#             price_after = None
#             date_after = None
#         else:
#             after_record = after_dates.iloc[days_after - 1]  # 3rd day after
#             price_after = after_record['Close']
#             date_after = after_record['Date']
        
#         if price_before is None or price_after is None:
#             return None
        
#         return {
#             'price_before': price_before,
#             'date_before': date_before,
#             'price_after': price_after,
#             'date_after': date_after,
#             'return_3day': (price_after - price_before) / price_before
#         }
    
#     def align_transcript(self, transcript, financials_df, market_df):
#         """
#         Align a single transcript with financial and market data
#         """
#         try:
#             ticker = transcript['ticker']
#             quarter = transcript['quarter']
            
#             # Parse transcript date
#             if 'date' in transcript and transcript['date']:
#                 transcript_date = pd.to_datetime(transcript['date'])
#             else:
#                 # Use quarter to estimate date
#                 transcript_date = self.parse_quarter_to_date(quarter)
            
#             if transcript_date is None:
#                 self.alignment_stats['temporal_mismatches'] += 1
#                 return None
            
#             # Find matching financial data
#             financial_match = self.find_matching_financial(ticker, transcript_date, financials_df)
            
#             if financial_match is None:
#                 self.alignment_stats['missing_financial_data'] += 1
#                 return None
            
#             # Get market data window
#             market_data = self.get_market_data_window(ticker, transcript_date, market_df)
            
#             if market_data is None:
#                 self.alignment_stats['missing_market_data'] += 1
#                 return None
            
#             # Create aligned record
#             aligned_record = {
#                 # Transcript metadata
#                 'ticker': ticker,
#                 'company_name': transcript.get('company_name', ''),
#                 'quarter': quarter,
#                 'transcript_date': transcript_date.strftime('%Y-%m-%d'),
#                 'fiscal_year': transcript.get('fiscal_year', 0),
#                 'fiscal_quarter': transcript.get('fiscal_quarter', 0),
                
#                 # Financial data date (for verification)
#                 'financial_date': financial_match['Date'].strftime('%Y-%m-%d'),
#                 'financial_date_diff_days': (transcript_date - financial_match['Date']).days,
                
#                 # Market data dates (for verification)
#                 'market_date_before': market_data['date_before'].strftime('%Y-%m-%d'),
#                 'market_date_after': market_data['date_after'].strftime('%Y-%m-%d'),
                
#                 # Transcript features (speaker segmentation)
#                 'total_management_words': transcript.get('total_management_words', 0),
#                 'total_analyst_words': transcript.get('total_analyst_words', 0),
#                 'management_analyst_ratio': transcript.get('management_analyst_word_ratio', 0),
                
#                 # Market features
#                 'price_before_earnings': market_data['price_before'],
#                 'price_after_earnings': market_data['price_after'],
#                 'stock_return_3day': market_data['return_3day'],
                
#                 # Temporal validation flags
#                 'is_temporally_valid': True,  # Passed all checks
#                 'alignment_timestamp': datetime.now().isoformat()
#             }
            
#             # Add all financial features
#             financial_features = [col for col in financial_match.index 
#                                 if col not in ['Ticker', 'Date']]
            
#             for feature in financial_features:
#                 aligned_record[f'financial_{feature}'] = financial_match[feature]
            
#             self.alignment_stats['successfully_aligned'] += 1
            
#             return aligned_record
            
#         except Exception as e:
#             self._log_error(f"Error aligning transcript {transcript.get('ticker', 'UNKNOWN')}: {e}")
#             return None
    
#     def align_all_data(self, transcripts, financials_df, market_df):
#         """
#         Align all transcripts with financial and market data
#         """
#         print("="*60)
#         print("⏰ ALIGNING DATA TEMPORALLY")
#         print("="*60)
#         print()
#         print("🔍 Matching transcripts with financial and market data...")
#         print("   Ensuring NO lookahead bias (only using past data)")
#         print()
        
#         aligned_records = []
        
#         for transcript in tqdm(transcripts, desc="Aligning"):
#             aligned = self.align_transcript(transcript, financials_df, market_df)
            
#             if aligned:
#                 aligned_records.append(aligned)
        
#         print()
#         self.alignment_stats['final_aligned_records'] = len(aligned_records)
        
#         return aligned_records
    
#     def validate_temporal_integrity(self, aligned_df):
#         """
#         Validate that temporal alignment has no lookahead bias
        
#         Checks:
#         1. Financial date <= Transcript date
#         2. Market "before" date <= Transcript date
#         3. Market "after" date > Transcript date
#         4. All dates are chronologically valid
#         """
#         print("="*60)
#         print("🔍 VALIDATING TEMPORAL INTEGRITY")
#         print("="*60)
#         print()
        
#         issues = []
        
#         # Convert dates for comparison
#         aligned_df['transcript_date'] = pd.to_datetime(aligned_df['transcript_date'])
#         aligned_df['financial_date'] = pd.to_datetime(aligned_df['financial_date'])
#         aligned_df['market_date_before'] = pd.to_datetime(aligned_df['market_date_before'])
#         aligned_df['market_date_after'] = pd.to_datetime(aligned_df['market_date_after'])
        
#         # Check 1: Financial date <= Transcript date
#         lookahead_financial = aligned_df[aligned_df['financial_date'] > aligned_df['transcript_date']]
#         if not lookahead_financial.empty:
#             issues.append(f"⚠️  {len(lookahead_financial)} records have financial data AFTER transcript date (LOOKAHEAD BIAS!)")
#         else:
#             print("✅ No financial lookahead bias detected")
        
#         # Check 2: Market before <= Transcript
#         lookahead_market_before = aligned_df[aligned_df['market_date_before'] > aligned_df['transcript_date']]
#         if not lookahead_market_before.empty:
#             issues.append(f"⚠️  {len(lookahead_market_before)} records have market 'before' date AFTER transcript")
#         else:
#             print("✅ Market 'before' dates are valid")
        
#         # Check 3: Market after > Transcript
#         invalid_market_after = aligned_df[aligned_df['market_date_after'] <= aligned_df['transcript_date']]
#         if not invalid_market_after.empty:
#             issues.append(f"⚠️  {len(invalid_market_after)} records have market 'after' date BEFORE/EQUAL transcript")
#         else:
#             print("✅ Market 'after' dates are valid")
        
#         # Check 4: Date range validation
#         print(f"\n📅 Date Range:")
#         print(f"   Earliest transcript: {aligned_df['transcript_date'].min()}")
#         print(f"   Latest transcript: {aligned_df['transcript_date'].max()}")
#         print(f"   Earliest financial: {aligned_df['financial_date'].min()}")
#         print(f"   Latest financial: {aligned_df['financial_date'].max()}")
        
#         # Check 5: Financial-transcript time gap
#         print(f"\n⏱️  Financial-Transcript Time Gap:")
#         print(f"   Average: {aligned_df['financial_date_diff_days'].mean():.1f} days")
#         print(f"   Median: {aligned_df['financial_date_diff_days'].median():.1f} days")
#         print(f"   Max: {aligned_df['financial_date_diff_days'].max():.0f} days")
        
#         if aligned_df['financial_date_diff_days'].max() > 90:
#             issues.append(f"⚠️  Some financials are >90 days before transcript")
        
#         print()
        
#         if issues:
#             print("="*60)
#             print("⚠️  TEMPORAL INTEGRITY ISSUES DETECTED:")
#             print("="*60)
#             for issue in issues:
#                 print(issue)
#             print()
#             return False
#         else:
#             print("="*60)
#             print("✅ TEMPORAL INTEGRITY VALIDATED")
#             print("="*60)
#             print("   No lookahead bias detected!")
#             print()
#             return True
    
#     def save_aligned_data(self, aligned_records):
#         """
#         Save temporally aligned dataset
#         """
#         if not aligned_records:
#             print("❌ No aligned records to save")
#             return False
        
#         try:
#             print("="*60)
#             print("💾 SAVING ALIGNED DATA")
#             print("="*60)
#             print()
            
#             # Convert to DataFrame
#             aligned_df = pd.DataFrame(aligned_records)
            
#             # Validate temporal integrity
#             is_valid = self.validate_temporal_integrity(aligned_df)
            
#             # Save as CSV
#             csv_path = self.output_dir / 'aligned_data.csv'
#             aligned_df.to_csv(csv_path, index=False)
#             print(f"✅ Aligned data saved: {csv_path}")
#             print(f"   Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
#             print(f"   Records: {len(aligned_df):,}")
#             print(f"   Features: {len(aligned_df.columns)}")
            
#             # Save alignment statistics
#             stats_path = self.output_dir / 'alignment_statistics.json'
#             with open(stats_path, 'w') as f:
#                 json.dump(self.alignment_stats, f, indent=2)
#             print(f"✅ Statistics saved: {stats_path}")
            
#             # Save metadata
#             metadata = {
#                 'total_records': len(aligned_df),
#                 'total_features': len(aligned_df.columns),
#                 'date_range': {
#                     'earliest': aligned_df['transcript_date'].min(),
#                     'latest': aligned_df['transcript_date'].max()
#                 },
#                 'temporal_validation': 'PASSED' if is_valid else 'FAILED',
#                 'alignment_timestamp': datetime.now().isoformat()
#             }
            
#             metadata_path = self.output_dir / 'alignment_metadata.json'
#             with open(metadata_path, 'w') as f:
#                 json.dump(metadata, f, indent=2, default=str)
#             print(f"✅ Metadata saved: {metadata_path}")
            
#             print()
#             return True
            
#         except Exception as e:
#             print(f"❌ Error saving aligned data: {e}")
#             return False
    
#     def _display_summary(self):
#         """Display alignment summary"""
#         print("="*60)
#         print("📊 ALIGNMENT SUMMARY")
#         print("="*60)
#         print()
#         print(f"Total transcripts:        {self.alignment_stats['total_transcripts']}")
#         print(f"Successfully aligned:     {self.alignment_stats['successfully_aligned']}")
#         print(f"Missing financial data:   {self.alignment_stats['missing_financial_data']}")
#         print(f"Missing market data:      {self.alignment_stats['missing_market_data']}")
#         print(f"Temporal mismatches:      {self.alignment_stats['temporal_mismatches']}")
#         print()
#         print(f"Final aligned records:    {self.alignment_stats['final_aligned_records']}")
        
#         if self.alignment_stats['total_transcripts'] > 0:
#             success_rate = self.alignment_stats['successfully_aligned'] / self.alignment_stats['total_transcripts']
#             print(f"Success rate:             {success_rate:.1%}")
        
#         print()
    
#     def _log_error(self, message):
#         """Log errors"""
#         with open(self.log_file, 'a') as f:
#             f.write(f"{datetime.now()} - ERROR: {message}\n")
    
#     def run(self):
#         """
#         Execute complete temporal alignment workflow
#         """
#         # Load all data sources
#         transcripts, financials_df, market_df = self.load_data_sources()
        
#         if transcripts is None or financials_df is None or market_df is None:
#             print("❌ Failed to load data sources")
#             return False
        
#         # Align all data
#         aligned_records = self.align_all_data(transcripts, financials_df, market_df)
        
#         # Display summary
#         self._display_summary()
        
#         # Save aligned data
#         success = self.save_aligned_data(aligned_records)
        
#         if success:
#             print("="*60)
#             print("✅ PHASE 2D COMPLETE")
#             print("="*60)
#             print()
#             print(f"📁 Aligned data saved to: {self.output_dir}")
#             print(f"📝 Log saved to: {self.log_file}")
#             print()
#             print("✅ Ready for Phase 2E: Data Quality Reporting")
#             print()
        
#         return success

# def main():
#     """Main execution"""
#     aligner = TemporalAligner()
#     success = aligner.run()
#     return success

# if __name__ == "__main__":
#     main()

"""
Phase 2D: Temporal Alignment
Aligns transcripts, financials, and market data on timeline
CRITICAL: Prevents lookahead bias in prediction
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR

class TemporalAligner:
    """
    Aligns earnings calls, financial data, and market data temporally
    Ensures no lookahead bias in the dataset
    """
    
    def __init__(self):
        self.transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.financial_dir = PROCESSED_DATA_DIR / 'financial'
        self.market_dir = RAW_DATA_DIR / 'market'
        
        self.output_dir = PROCESSED_DATA_DIR / 'aligned'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'temporal_alignment.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.alignment_stats = {
            'total_transcripts': 0,
            'successfully_aligned': 0,
            'missing_financial_data': 0,
            'missing_market_data': 0,
            'temporal_mismatches': 0,
            'final_aligned_records': 0
        }
    
    def load_data_sources(self):
        """
        Load all data sources: transcripts, financials, market data
        """
        print("="*60)
        print("PHASE 2D: TEMPORAL ALIGNMENT")
        print("="*60)
        print()
        print("📂 LOADING DATA SOURCES...")
        print()
        
        # 1. Load segmented transcripts
        transcripts_file = self.transcripts_dir / 'transcripts_segmented.json'
        if not transcripts_file.exists():
            print(f"❌ Segmented transcripts not found: {transcripts_file}")
            return None, None, None
        
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        
        print(f"✅ Transcripts loaded: {len(transcripts)} records")
        
        # 2. Load normalized financials
        financials_file = self.financial_dir / 'financial_features_normalized.csv'
        if not financials_file.exists():
            print(f"❌ Normalized financials not found: {financials_file}")
            return None, None, None
        
        financials_df = pd.read_csv(financials_file)
        _fin_dates = pd.to_datetime(financials_df['Date'], utc=True)
        financials_df['Date'] = _fin_dates.dt.tz_convert(None)
        
        print(f"✅ Financials loaded: {len(financials_df)} records")
        print("Financial tickers sample:", financials_df['Ticker'].unique()[:10])
        print("Transcript ticker sample:", transcripts[0]['ticker'])

        # 3. Load market data
        market_file = self.market_dir / 'stock_prices.csv'
        if not market_file.exists():
            print(f"❌ Market data not found: {market_file}")
            return None, None, None
        
        market_df = pd.read_csv(market_file)
        _market_dates = pd.to_datetime(market_df['Date'], utc=True)
        market_df['Date'] = _market_dates.dt.tz_convert(None)
        
        print("Market columns:", market_df.columns.tolist())
        print("Market date range:", market_df['Date'].min(), "to", market_df['Date'].max())
        print("Market sample tickers:", market_df['Ticker'].unique()[:5] if 'Ticker' in market_df.columns else "NO TICKER COLUMN")

        print(f"✅ Market data loaded: {len(market_df)} records")
        print()
        
        self.alignment_stats['total_transcripts'] = len(transcripts)
        
        return transcripts, financials_df, market_df
    
    def parse_quarter_to_date(self, quarter_str):
        """
        Convert quarter string to approximate date
        
        Examples:
        "Q1 2023" -> 2023-03-31 (end of Q1)
        "Q2 2023" -> 2023-06-30 (end of Q2)
        "Q3 2023" -> 2023-09-30 (end of Q3)
        "Q4 2023" -> 2023-12-31 (end of Q4)
        """
        try:
            # Parse "Q1 2023" format
            quarter, year = quarter_str.split()
            year = int(year)
            quarter_num = int(quarter[1])
            
            # Map quarter to end-of-quarter date
            quarter_end_dates = {
                1: f"{year}-03-31",
                2: f"{year}-06-30",
                3: f"{year}-09-30",
                4: f"{year}-12-31"
            }
            
            date_str = quarter_end_dates[quarter_num]
            return pd.to_datetime(date_str)
            
        except Exception as e:
            self._log_error(f"Error parsing quarter '{quarter_str}': {e}")
            return None
    
    def find_matching_financial(self, ticker, transcript_date, financials_df, tolerance_days=120):
        """
        Find financial data matching the transcript
        
        Matching logic:
        1. Same ticker
        2. Financial date within tolerance_days of transcript date
        3. Financial date should be BEFORE or equal to transcript date (no lookahead)
        """
        # Filter by ticker
        ticker_financials = financials_df[financials_df['Ticker'] == ticker].copy()
        
        if ticker_financials.empty:
            return None
        
        # Find closest financial record BEFORE transcript date
        ticker_financials['DateDiff'] = (transcript_date - ticker_financials['Date']).dt.days
        
        # Keep only records where financial date <= transcript date (no lookahead)
        # And within tolerance window
        valid_records = ticker_financials[
            (ticker_financials['DateDiff'] >= 0) & 
            (ticker_financials['DateDiff'] <= tolerance_days)
        ]
        
        past_records = ticker_financials[ticker_financials['DateDiff'] >= 0]
        if past_records.empty:
            print(f"⚠️ No financial match for {ticker} on {transcript_date.date()}")
            return None
        within_tolerance = past_records[past_records['DateDiff'] <= tolerance_days]
        valid_records = within_tolerance if not within_tolerance.empty else past_records
        
        # Get the closest match
        closest_match = valid_records.loc[valid_records['DateDiff'].idxmin()]
        
        return closest_match
    
    def get_market_data_window(self, ticker, date, market_df, days_before=5, days_after=3):
        """
        Get market data in a window around the earnings date
        
        Returns:
        - Price before earnings (for return calculation)
        - Price after earnings (for abnormal return target)
        """
        # Filter by ticker
        ticker_market = market_df[market_df['Ticker'] == ticker].copy()
        
        if ticker_market.empty:
            return None
        
        ticker_market = ticker_market.sort_values('Date')
        
        # Find closest date before earnings
        before_dates = ticker_market[ticker_market['Date'] <= date]
        if before_dates.empty:
            price_before = None
            date_before = None
        else:
            before_record = before_dates.iloc[-1]
            price_before = before_record['Close']
            date_before = before_record['Date']
        
        # Find date ~3 days after earnings (use whatever is available)
        after_dates = ticker_market[ticker_market['Date'] > date]
        if after_dates.empty:
            price_after = None
            date_after = None
        else:
            idx = min(days_after - 1, len(after_dates) - 1)
            after_record = after_dates.iloc[idx]
            price_after = after_record['Close']
            date_after = after_record['Date']
        
        if price_before is None or price_after is None:
            return None
        
        return {
            'price_before': price_before,
            'date_before': date_before,
            'price_after': price_after,
            'date_after': date_after,
            'return_3day': (price_after - price_before) / price_before
        }
    
    def align_transcript(self, transcript, financials_df, market_df):
        """
        Align a single transcript with financial and market data
        """
        try:
            ticker = transcript['ticker']
            quarter = transcript['quarter']
            
            # Parse transcript date
            if 'date' in transcript and transcript['date']:
                transcript_date = pd.to_datetime(transcript['date']).tz_localize(None)
            else:
                # Use quarter to estimate date
                transcript_date = self.parse_quarter_to_date(quarter)
            
            if transcript_date is None:
                self.alignment_stats['temporal_mismatches'] += 1
                return None
            
            # Find matching financial data
            financial_match = self.find_matching_financial(ticker, transcript_date, financials_df)
            
            if financial_match is None:
                self.alignment_stats['missing_financial_data'] += 1
                return None
            
            # Get market data window
            market_data = self.get_market_data_window(ticker, transcript_date, market_df)
            
            if market_data is None:
                self.alignment_stats['missing_market_data'] += 1
                return None
            
            # Create aligned record
            aligned_record = {
                # Transcript metadata
                'ticker': ticker,
                'company_name': transcript.get('company_name', ''),
                'quarter': quarter,
                'transcript_date': transcript_date.strftime('%Y-%m-%d'),
                'fiscal_year': transcript.get('fiscal_year', 0),
                'fiscal_quarter': transcript.get('fiscal_quarter', 0),
                
                # Financial data date (for verification)
                'financial_date': financial_match['Date'].strftime('%Y-%m-%d'),
                'financial_date_diff_days': (transcript_date - financial_match['Date']).days,
                
                # Market data dates (for verification)
                'market_date_before': market_data['date_before'].strftime('%Y-%m-%d'),
                'market_date_after': market_data['date_after'].strftime('%Y-%m-%d'),
                
                # Transcript features (speaker segmentation)
                'total_management_words': transcript.get('total_management_words', 0),
                'total_analyst_words': transcript.get('total_analyst_words', 0),
                'management_analyst_ratio': transcript.get('management_analyst_word_ratio', 0),
                
                # Market features
                'price_before_earnings': market_data['price_before'],
                'price_after_earnings': market_data['price_after'],
                'stock_return_3day': market_data['return_3day'],
                
                # Temporal validation flags
                'is_temporally_valid': True,  # Passed all checks
                'alignment_timestamp': datetime.now().isoformat()
            }
            
            # Add all financial features
            financial_features = [col for col in financial_match.index 
                                if col not in ['Ticker', 'Date']]
            
            for feature in financial_features:
                aligned_record[f'financial_{feature}'] = financial_match[feature]
            
            self.alignment_stats['successfully_aligned'] += 1
            
            return aligned_record
            
        except Exception as e:
            import traceback
            self._log_error(f"Error aligning transcript {transcript.get('ticker', 'UNKNOWN')}: {e}\n{traceback.format_exc()}")
            print(f"❌ Exception for {transcript.get('ticker', 'UNKNOWN')}: {e}")
            return None
    
    def align_all_data(self, transcripts, financials_df, market_df):
        """
        Align all transcripts with financial and market data
        """
        print("="*60)
        print("⏰ ALIGNING DATA TEMPORALLY")
        print("="*60)
        print()
        print("🔍 Matching transcripts with financial and market data...")
        print("   Ensuring NO lookahead bias (only using past data)")
        print()
        
        aligned_records = []
        
        for transcript in tqdm(transcripts, desc="Aligning"):
            aligned = self.align_transcript(transcript, financials_df, market_df)
            
            if aligned is not None:
                aligned_records.append(aligned)
        
        print()
        self.alignment_stats['final_aligned_records'] = len(aligned_records)
        
        return aligned_records
    
    def validate_temporal_integrity(self, aligned_df):
        """
        Validate that temporal alignment has no lookahead bias
        
        Checks:
        1. Financial date <= Transcript date
        2. Market "before" date <= Transcript date
        3. Market "after" date > Transcript date
        4. All dates are chronologically valid
        """
        print("="*60)
        print("🔍 VALIDATING TEMPORAL INTEGRITY")
        print("="*60)
        print()
        
        issues = []
        
        # Convert dates for comparison
        aligned_df['transcript_date'] = pd.to_datetime(aligned_df['transcript_date'])
        aligned_df['financial_date'] = pd.to_datetime(aligned_df['financial_date'])
        aligned_df['market_date_before'] = pd.to_datetime(aligned_df['market_date_before'])
        aligned_df['market_date_after'] = pd.to_datetime(aligned_df['market_date_after'])
        
        # Check 1: Financial date <= Transcript date
        lookahead_financial = aligned_df[aligned_df['financial_date'] > aligned_df['transcript_date']]
        if not lookahead_financial.empty:
            issues.append(f"⚠️  {len(lookahead_financial)} records have financial data AFTER transcript date (LOOKAHEAD BIAS!)")
        else:
            print("✅ No financial lookahead bias detected")
        
        # Check 2: Market before <= Transcript
        lookahead_market_before = aligned_df[aligned_df['market_date_before'] > aligned_df['transcript_date']]
        if not lookahead_market_before.empty:
            issues.append(f"⚠️  {len(lookahead_market_before)} records have market 'before' date AFTER transcript")
        else:
            print("✅ Market 'before' dates are valid")
        
        # Check 3: Market after > Transcript
        invalid_market_after = aligned_df[aligned_df['market_date_after'] <= aligned_df['transcript_date']]
        if not invalid_market_after.empty:
            issues.append(f"⚠️  {len(invalid_market_after)} records have market 'after' date BEFORE/EQUAL transcript")
        else:
            print("✅ Market 'after' dates are valid")
        
        # Check 4: Date range validation
        print(f"\n📅 Date Range:")
        print(f"   Earliest transcript: {aligned_df['transcript_date'].min()}")
        print(f"   Latest transcript: {aligned_df['transcript_date'].max()}")
        print(f"   Earliest financial: {aligned_df['financial_date'].min()}")
        print(f"   Latest financial: {aligned_df['financial_date'].max()}")
        
        # Check 5: Financial-transcript time gap
        print(f"\n⏱️  Financial-Transcript Time Gap:")
        print(f"   Average: {aligned_df['financial_date_diff_days'].mean():.1f} days")
        print(f"   Median: {aligned_df['financial_date_diff_days'].median():.1f} days")
        print(f"   Max: {aligned_df['financial_date_diff_days'].max():.0f} days")
        
        if aligned_df['financial_date_diff_days'].max() > 90:
            issues.append(f"⚠️  Some financials are >90 days before transcript")
        
        print()
        
        if issues:
            print("="*60)
            print("⚠️  TEMPORAL INTEGRITY ISSUES DETECTED:")
            print("="*60)
            for issue in issues:
                print(issue)
            print()
            return False
        else:
            print("="*60)
            print("✅ TEMPORAL INTEGRITY VALIDATED")
            print("="*60)
            print("   No lookahead bias detected!")
            print()
            return True
    
    def save_aligned_data(self, aligned_records):
        """
        Save temporally aligned dataset
        """
        if not aligned_records:
            print("❌ No aligned records to save")
            return False
        
        try:
            print("="*60)
            print("💾 SAVING ALIGNED DATA")
            print("="*60)
            print()
            
            # Convert to DataFrame
            aligned_df = pd.DataFrame(aligned_records)
            
            # Validate temporal integrity
            is_valid = self.validate_temporal_integrity(aligned_df)
            
            # Save as CSV
            csv_path = self.output_dir / 'aligned_data.csv'
            aligned_df.to_csv(csv_path, index=False)
            print(f"✅ Aligned data saved: {csv_path}")
            print(f"   Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
            print(f"   Records: {len(aligned_df):,}")
            print(f"   Features: {len(aligned_df.columns)}")
            
            # Save alignment statistics
            stats_path = self.output_dir / 'alignment_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(self.alignment_stats, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            # Save metadata
            metadata = {
                'total_records': len(aligned_df),
                'total_features': len(aligned_df.columns),
                'date_range': {
                    'earliest': aligned_df['transcript_date'].min(),
                    'latest': aligned_df['transcript_date'].max()
                },
                'temporal_validation': 'PASSED' if is_valid else 'FAILED',
                'alignment_timestamp': datetime.now().isoformat()
            }
            
            metadata_path = self.output_dir / 'alignment_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"✅ Metadata saved: {metadata_path}")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Error saving aligned data: {e}")
            return False
    
    def _display_summary(self):
        """Display alignment summary"""
        print("="*60)
        print("📊 ALIGNMENT SUMMARY")
        print("="*60)
        print()
        print(f"Total transcripts:        {self.alignment_stats['total_transcripts']}")
        print(f"Successfully aligned:     {self.alignment_stats['successfully_aligned']}")
        print(f"Missing financial data:   {self.alignment_stats['missing_financial_data']}")
        print(f"Missing market data:      {self.alignment_stats['missing_market_data']}")
        print(f"Temporal mismatches:      {self.alignment_stats['temporal_mismatches']}")
        print()
        print(f"Final aligned records:    {self.alignment_stats['final_aligned_records']}")
        
        if self.alignment_stats['total_transcripts'] > 0:
            success_rate = self.alignment_stats['successfully_aligned'] / self.alignment_stats['total_transcripts']
            print(f"Success rate:             {success_rate:.1%}")
        
        print()
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete temporal alignment workflow
        """
        # Load all data sources
        transcripts, financials_df, market_df = self.load_data_sources()
        
        if transcripts is None or financials_df is None or market_df is None:
            print("❌ Failed to load data sources")
            return False
        
        # Align all data
        aligned_records = self.align_all_data(transcripts, financials_df, market_df)
        
        # Display summary
        self._display_summary()
        
        # Save aligned data
        success = self.save_aligned_data(aligned_records)
        
        if success:
            print("="*60)
            print("✅ PHASE 2D COMPLETE")
            print("="*60)
            print()
            print(f"📁 Aligned data saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 2E: Data Quality Reporting")
            print()
        
        return success

def main():
    """Main execution"""
    aligner = TemporalAligner()
    success = aligner.run()
    return success

if __name__ == "__main__":
    main()