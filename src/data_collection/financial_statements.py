"""
Phase 1C: Financial Statements Collection
Collects quarterly financial data for S&P 500 companies
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, LOGS_DIR, START_DATE, END_DATE

class FinancialStatementsCollector:
    """
    Collects quarterly financial statements from Yahoo Finance
    """
    
    def __init__(self, tickers):
        self.tickers = tickers
        
        self.output_dir = RAW_DATA_DIR / 'financial'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'data_collection' / 'financial_statements.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.failed_tickers = []
        self.successful_tickers = []
        
        # Storage for collected data
        self.all_income_statements = []
        self.all_balance_sheets = []
        self.all_cashflows = []
    
    def collect_financials_for_ticker(self, ticker):
        """
        Collect all financial statements for a single ticker
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financial statements
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            # Check if data exists
            if income_stmt is None or income_stmt.empty:
                return None, None, None
            
            # Transpose to have dates as rows
            income_stmt = income_stmt.T
            balance_sheet = balance_sheet.T if balance_sheet is not None else pd.DataFrame()
            cashflow = cashflow.T if cashflow is not None else pd.DataFrame()
            
            # Add ticker column
            income_stmt['Ticker'] = ticker
            balance_sheet['Ticker'] = ticker
            cashflow['Ticker'] = ticker
            
            # Reset index to make dates a column
            income_stmt.reset_index(inplace=True)
            balance_sheet.reset_index(inplace=True)
            cashflow.reset_index(inplace=True)
            
            # Rename index column
            income_stmt.rename(columns={'index': 'Date'}, inplace=True)
            balance_sheet.rename(columns={'index': 'Date'}, inplace=True)
            cashflow.rename(columns={'index': 'Date'}, inplace=True)
            
            return income_stmt, balance_sheet, cashflow
            
        except Exception as e:
            self._log_error(f"{ticker}: {e}")
            return None, None, None
    
    def collect_all_financials(self):
        """
        Collect financial statements for all tickers
        """
        print("="*60)
        print("PHASE 1C: FINANCIAL STATEMENTS COLLECTION")
        print("="*60)
        print()
        print(f"📊 Tickers to process: {len(self.tickers)}")
        print(f"📋 Collecting: Income Statements, Balance Sheets, Cash Flows")
        print()
        
        # Progress bar
        for ticker in tqdm(self.tickers, desc="Collecting financial statements"):
            income, balance, cashflow = self.collect_financials_for_ticker(ticker)
            
            if income is not None and not income.empty:
                self.all_income_statements.append(income)
                self.all_balance_sheets.append(balance)
                self.all_cashflows.append(cashflow)
                self.successful_tickers.append(ticker)
            else:
                self.failed_tickers.append(ticker)
            
            # Rate limiting
            time.sleep(0.15)
        
        # Combine all data
        print()
        print("="*60)
        print("📊 COMBINING DATA...")
        print("="*60)
        
        combined_income = None
        combined_balance = None
        combined_cashflow = None
        
        if self.all_income_statements:
            combined_income = pd.concat(self.all_income_statements, ignore_index=True)
            print(f"✅ Income Statements: {len(combined_income):,} records")
        
        if self.all_balance_sheets:
            combined_balance = pd.concat(self.all_balance_sheets, ignore_index=True)
            print(f"✅ Balance Sheets: {len(combined_balance):,} records")
        
        if self.all_cashflows:
            combined_cashflow = pd.concat(self.all_cashflows, ignore_index=True)
            print(f"✅ Cash Flows: {len(combined_cashflow):,} records")
        
        print()
        print("="*60)
        print("📊 COLLECTION SUMMARY")
        print("="*60)
        print(f"✅ Successful: {len(self.successful_tickers)}/{len(self.tickers)}")
        print(f"❌ Failed: {len(self.failed_tickers)}/{len(self.tickers)}")
        
        if self.failed_tickers:
            print(f"\n⚠️  Failed tickers: {', '.join(self.failed_tickers[:10])}")
            if len(self.failed_tickers) > 10:
                print(f"   ... and {len(self.failed_tickers) - 10} more")
        
        print()
        
        return combined_income, combined_balance, combined_cashflow
    
    def extract_key_features(self, income_df, balance_df, cashflow_df):
        """
        Extract the 28 key financial features mentioned in the proposal
        """
        print("="*60)
        print("📊 EXTRACTING KEY FINANCIAL FEATURES")
        print("="*60)
        print()
        
        # Create a master dataframe
        features_list = []
        
        # Get unique ticker-date combinations
        if income_df is not None:
            for ticker in income_df['Ticker'].unique():
                ticker_income = income_df[income_df['Ticker'] == ticker]
                ticker_balance = balance_df[balance_df['Ticker'] == ticker] if balance_df is not None else pd.DataFrame()
                ticker_cashflow = cashflow_df[cashflow_df['Ticker'] == ticker] if cashflow_df is not None else pd.DataFrame()
                
                for idx, row in ticker_income.iterrows():
                    date = row['Date']
                    
                    # Find matching balance sheet and cashflow
                    balance_row = ticker_balance[ticker_balance['Date'] == date]
                    cashflow_row = ticker_cashflow[ticker_cashflow['Date'] == date]
                    
                    # Extract features
                    feature_dict = {
                        'Ticker': ticker,
                        'Date': date,
                        
                        # Income Statement Features (10)
                        'Total_Revenue': row.get('Total Revenue', None),
                        'Gross_Profit': row.get('Gross Profit', None),
                        'Operating_Income': row.get('Operating Income', None),
                        'Net_Income': row.get('Net Income', None),
                        'EBIT': row.get('EBIT', None),
                        'EBITDA': row.get('EBITDA', None),
                        'Operating_Expense': row.get('Operating Expense', None),
                        'Interest_Expense': row.get('Interest Expense', None),
                        'Tax_Provision': row.get('Tax Provision', None),
                        'Diluted_EPS': row.get('Diluted EPS', None),
                        
                        # Balance Sheet Features (10)
                        'Total_Assets': balance_row['Total Assets'].values[0] if not balance_row.empty and 'Total Assets' in balance_row.columns else None,
                        'Total_Liabilities': balance_row['Total Liabilities Net Minority Interest'].values[0] if not balance_row.empty and 'Total Liabilities Net Minority Interest' in balance_row.columns else None,
                        'Total_Equity': balance_row['Total Equity Gross Minority Interest'].values[0] if not balance_row.empty and 'Total Equity Gross Minority Interest' in balance_row.columns else None,
                        'Current_Assets': balance_row['Current Assets'].values[0] if not balance_row.empty and 'Current Assets' in balance_row.columns else None,
                        'Current_Liabilities': balance_row['Current Liabilities'].values[0] if not balance_row.empty and 'Current Liabilities' in balance_row.columns else None,
                        'Cash_And_Equivalents': balance_row['Cash And Cash Equivalents'].values[0] if not balance_row.empty and 'Cash And Cash Equivalents' in balance_row.columns else None,
                        'Total_Debt': balance_row['Total Debt'].values[0] if not balance_row.empty and 'Total Debt' in balance_row.columns else None,
                        'Retained_Earnings': balance_row['Retained Earnings'].values[0] if not balance_row.empty and 'Retained Earnings' in balance_row.columns else None,
                        'Working_Capital': balance_row['Working Capital'].values[0] if not balance_row.empty and 'Working Capital' in balance_row.columns else None,
                        'Inventories': balance_row['Inventory'].values[0] if not balance_row.empty and 'Inventory' in balance_row.columns else None,
                        
                        # Cash Flow Features (8)
                        'Operating_Cash_Flow': cashflow_row['Operating Cash Flow'].values[0] if not cashflow_row.empty and 'Operating Cash Flow' in cashflow_row.columns else None,
                        'Investing_Cash_Flow': cashflow_row['Investing Cash Flow'].values[0] if not cashflow_row.empty and 'Investing Cash Flow' in cashflow_row.columns else None,
                        'Financing_Cash_Flow': cashflow_row['Financing Cash Flow'].values[0] if not cashflow_row.empty and 'Financing Cash Flow' in cashflow_row.columns else None,
                        'Free_Cash_Flow': cashflow_row['Free Cash Flow'].values[0] if not cashflow_row.empty and 'Free Cash Flow' in cashflow_row.columns else None,
                        'Capital_Expenditure': cashflow_row['Capital Expenditure'].values[0] if not cashflow_row.empty and 'Capital Expenditure' in cashflow_row.columns else None,
                        'Dividend_Paid': cashflow_row['Cash Dividends Paid'].values[0] if not cashflow_row.empty and 'Cash Dividends Paid' in cashflow_row.columns else None,
                        'Repurchase_Of_Stock': cashflow_row['Repurchase Of Capital Stock'].values[0] if not cashflow_row.empty and 'Repurchase Of Capital Stock' in cashflow_row.columns else None,
                        'Change_In_Cash': cashflow_row['Changes In Cash'].values[0] if not cashflow_row.empty and 'Changes In Cash' in cashflow_row.columns else None,
                    }
                    
                    features_list.append(feature_dict)
        
        features_df = pd.DataFrame(features_list)
        
        print(f"✅ Extracted features for {len(features_df):,} ticker-quarter combinations")
        print(f"✅ Total features: {len(features_df.columns) - 2} (excluding Ticker and Date)")
        print()
        
        # Show feature completeness
        print("📊 Feature Completeness:")
        completeness = (features_df.notna().sum() / len(features_df) * 100).round(1)
        for col in features_df.columns:
            if col not in ['Ticker', 'Date']:
                print(f"   {col}: {completeness[col]:.1f}%")
        
        print()
        
        return features_df
    
    def save_data(self, income_df, balance_df, cashflow_df, features_df):
        """
        Save all financial data
        """
        try:
            print("="*60)
            print("💾 SAVING DATA")
            print("="*60)
            print()
            
            # Save raw statements
            if income_df is not None:
                income_path = self.output_dir / 'income_statements.csv'
                income_df.to_csv(income_path, index=False)
                print(f"✅ Income statements saved: {income_path}")
                print(f"   Size: {income_path.stat().st_size / (1024*1024):.2f} MB")
            
            if balance_df is not None:
                balance_path = self.output_dir / 'balance_sheets.csv'
                balance_df.to_csv(balance_path, index=False)
                print(f"✅ Balance sheets saved: {balance_path}")
                print(f"   Size: {balance_path.stat().st_size / (1024*1024):.2f} MB")
            
            if cashflow_df is not None:
                cashflow_path = self.output_dir / 'cashflows.csv'
                cashflow_df.to_csv(cashflow_path, index=False)
                print(f"✅ Cash flows saved: {cashflow_path}")
                print(f"   Size: {cashflow_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save extracted features
            if features_df is not None:
                features_path = self.output_dir / 'financial_features.csv'
                features_df.to_csv(features_path, index=False)
                print(f"✅ Financial features saved: {features_path}")
                print(f"   Size: {features_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save collection log
            log_data = {
                'Successful_Tickers': self.successful_tickers,
                'Failed_Tickers': self.failed_tickers
            }
            log_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in log_data.items()]))
            log_path = self.output_dir / 'collection_log.csv'
            log_df.to_csv(log_path, index=False)
            print(f"✅ Collection log saved: {log_path}")
            
            print()
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
        Execute complete financial statements collection workflow
        """
        # Collect all financial statements
        income_df, balance_df, cashflow_df = self.collect_all_financials()
        
        # Extract key features
        features_df = None
        if income_df is not None:
            features_df = self.extract_key_features(income_df, balance_df, cashflow_df)
        
        # Save data
        if income_df is not None:
            success = self.save_data(income_df, balance_df, cashflow_df, features_df)
            
            if success:
                print("="*60)
                print("✅ PHASE 1C COMPLETE")
                print("="*60)
                print(f"\n📁 Data saved to: {self.output_dir}")
                
                # Calculate success rate
                success_rate = len(self.successful_tickers) / len(self.tickers)
                print(f"\n📊 Success Rate: {success_rate:.1%}")
                
                if success_rate >= 0.85:
                    print("✅ Success rate meets threshold (≥85%)")
                else:
                    print("⚠️  Success rate below 85% threshold")
                
                print("\n✅ Ready for Phase 1D: Earnings Call Transcripts Collection")
                
                return income_df, balance_df, cashflow_df, features_df
        
        return None, None, None, None

def main():
    """Main execution"""
    # Load tickers
    ticker_path = RAW_DATA_DIR / 'market' / 'sp500_tickers.csv'
    
    if not ticker_path.exists():
        print("❌ Ticker file not found. Please run Phase 1A first.")
        return
    
    tickers_df = pd.read_csv(ticker_path)
    tickers = tickers_df['Ticker'].tolist()
    
    # Collect financial statements
    collector = FinancialStatementsCollector(tickers)
    income, balance, cashflow, features = collector.run()
    
    return income, balance, cashflow, features

if __name__ == "__main__":
    main()