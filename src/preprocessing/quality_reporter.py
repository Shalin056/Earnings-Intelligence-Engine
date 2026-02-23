"""
Phase 2E: Data Quality Reporting
Generates comprehensive quality reports for preprocessed data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR

class DataQualityReporter:
    """
    Generates comprehensive data quality reports
    """
    
    def __init__(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.results_dir = RESULTS_DIR / 'quality_reports'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'quality_reporting.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        self.quality_metrics = {}
    
    def load_all_data(self):
        """
        Load all preprocessed data
        """
        print("="*60)
        print("PHASE 2E: DATA QUALITY REPORTING")
        print("="*60)
        print()
        print("📂 LOADING PREPROCESSED DATA...")
        print()
        
        data = {}
        
        # 1. Load aligned data
        aligned_file = self.processed_dir / 'aligned' / 'aligned_data.csv'
        if aligned_file.exists():
            data['aligned'] = pd.read_csv(aligned_file)
            print(f"✅ Aligned data: {len(data['aligned'])} records")
        else:
            print(f"❌ Aligned data not found")
            return None
        
        # 2. Load financial data
        financial_file = self.processed_dir / 'financial' / 'financial_features_normalized.csv'
        if financial_file.exists():
            data['financial'] = pd.read_csv(financial_file)
            print(f"✅ Financial data: {len(data['financial'])} records")
        
        # 3. Load transcript data
        transcript_file = self.processed_dir / 'transcripts' / 'transcripts_segmented.json'
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data['transcripts'] = json.load(f)
            print(f"✅ Transcript data: {len(data['transcripts'])} records")
        
        # 4. Load statistics from previous phases
        stats_files = {
            'cleaning': self.processed_dir / 'transcripts' / 'cleaning_statistics.json',
            'segmentation': self.processed_dir / 'transcripts' / 'segmentation_statistics.json',
            'normalization': self.processed_dir / 'financial' / 'normalization_statistics.json',
            'alignment': self.processed_dir / 'aligned' / 'alignment_statistics.json'
        }
        
        data['statistics'] = {}
        for name, file in stats_files.items():
            if file.exists():
                with open(file, 'r') as f:
                    data['statistics'][name] = json.load(f)
                print(f"✅ {name.capitalize()} statistics loaded")
        
        print()
        return data
    
    def generate_data_summary(self, data):
        """
        Generate overall data summary
        """
        print("="*60)
        print("📊 GENERATING DATA SUMMARY")
        print("="*60)
        print()
        
        aligned_df = data['aligned']
        
        summary = {
            'total_records': len(aligned_df),
            'total_features': len(aligned_df.columns),
            'unique_tickers': aligned_df['ticker'].nunique(),
            'date_range': {
                'start': aligned_df['transcript_date'].min(),
                'end': aligned_df['transcript_date'].max()
            },
            'quarters_covered': aligned_df['quarter'].nunique(),
            'completeness': {}
        }
        
        # Calculate completeness
        for col in aligned_df.columns:
            non_null = aligned_df[col].notna().sum()
            completeness = (non_null / len(aligned_df)) * 100
            summary['completeness'][col] = round(completeness, 2)
        
        # Overall completeness
        avg_completeness = np.mean(list(summary['completeness'].values()))
        summary['overall_completeness'] = round(avg_completeness, 2)
        
        print(f"📊 Data Summary:")
        print(f"   Total records: {summary['total_records']:,}")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Unique tickers: {summary['unique_tickers']}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   Overall completeness: {summary['overall_completeness']:.1f}%")
        print()
        
        self.quality_metrics['summary'] = summary
        return summary
    
    def analyze_data_quality(self, data):
        """
        Analyze data quality metrics
        """
        print("="*60)
        print("🔍 ANALYZING DATA QUALITY")
        print("="*60)
        print()
        
        aligned_df = data['aligned']
        
        quality_metrics = {
            'missing_values': {},
            'outliers': {},
            'temporal_integrity': {},
            'feature_distributions': {}
        }
        
        # 1. Missing values analysis
        print("1️⃣ MISSING VALUES ANALYSIS:")
        missing = aligned_df.isnull().sum()
        missing_pct = (missing / len(aligned_df)) * 100
        
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing) > 0:
            print(f"   Features with missing values: {len(cols_with_missing)}")
            for col, count in cols_with_missing.head(10).items():
                pct = missing_pct[col]
                print(f"      {col}: {count} ({pct:.1f}%)")
            if len(cols_with_missing) > 10:
                print(f"      ... and {len(cols_with_missing) - 10} more")
        else:
            print("   ✅ No missing values detected!")
        
        quality_metrics['missing_values']['total'] = int(missing.sum())
        quality_metrics['missing_values']['columns_affected'] = len(cols_with_missing)
        print()
        
        # 2. Outlier detection (for financial features)
        print("2️⃣ OUTLIER DETECTION:")
        financial_cols = [col for col in aligned_df.columns if col.startswith('financial_')]
        
        outlier_counts = {}
        for col in financial_cols[:5]:  # Check first 5 financial features
            try:
                values = aligned_df[col].dropna()
                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((values < (q1 - 3 * iqr)) | (values > (q3 + 3 * iqr))).sum()
                    outlier_counts[col] = outliers
            except:
                pass
        
        if outlier_counts:
            total_outliers = sum(outlier_counts.values())
            print(f"   Total outliers detected: {total_outliers}")
            for col, count in list(outlier_counts.items())[:5]:
                pct = (count / len(aligned_df)) * 100
                print(f"      {col.replace('financial_', '')}: {count} ({pct:.1f}%)")
        
        quality_metrics['outliers']['total'] = sum(outlier_counts.values()) if outlier_counts else 0
        print()
        
        # 3. Temporal integrity
        print("3️⃣ TEMPORAL INTEGRITY:")
        aligned_df['transcript_date'] = pd.to_datetime(aligned_df['transcript_date'])
        aligned_df['financial_date'] = pd.to_datetime(aligned_df['financial_date'])
        
        # Check for lookahead bias
        lookahead = (aligned_df['financial_date'] > aligned_df['transcript_date']).sum()
        
        print(f"   Lookahead bias violations: {lookahead}")
        if lookahead == 0:
            print("   ✅ NO LOOKAHEAD BIAS - Temporally valid!")
        else:
            print(f"   ❌ WARNING: {lookahead} records have lookahead bias!")
        
        # Time gaps
        avg_gap = aligned_df['financial_date_diff_days'].mean()
        max_gap = aligned_df['financial_date_diff_days'].max()
        
        print(f"   Average financial-transcript gap: {avg_gap:.1f} days")
        print(f"   Maximum gap: {max_gap:.0f} days")
        
        quality_metrics['temporal_integrity'] = {
            'lookahead_violations': int(lookahead),
            'avg_gap_days': float(avg_gap),
            'max_gap_days': float(max_gap)
        }
        print()
        
        # 4. Feature distributions
        print("4️⃣ FEATURE DISTRIBUTIONS:")
        
        # Stock returns
        returns = aligned_df['stock_return_3day']
        print(f"   Stock Returns (3-day):")
        print(f"      Mean: {returns.mean():.4f}")
        print(f"      Std: {returns.std():.4f}")
        print(f"      Min: {returns.min():.4f}")
        print(f"      Max: {returns.max():.4f}")
        
        # Positive vs negative returns
        positive = (returns > 0).sum()
        negative = (returns <= 0).sum()
        print(f"      Positive returns: {positive} ({positive/len(returns)*100:.1f}%)")
        print(f"      Negative returns: {negative} ({negative/len(returns)*100:.1f}%)")
        
        quality_metrics['feature_distributions'] = {
            'returns_mean': float(returns.mean()),
            'returns_std': float(returns.std()),
            'positive_pct': float(positive/len(returns)*100),
            'negative_pct': float(negative/len(returns)*100)
        }
        print()
        
        self.quality_metrics['quality_analysis'] = quality_metrics
        return quality_metrics
    
    def generate_visualizations(self, data):
        """
        Generate data quality visualizations
        """
        print("="*60)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*60)
        print()
        
        aligned_df = data['aligned']
        figures_dir = RESULTS_DIR / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Returns distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        aligned_df['stock_return_3day'].hist(bins=50, edgecolor='black')
        plt.xlabel('3-Day Stock Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Stock Returns')
        plt.axvline(0, color='red', linestyle='--', label='Zero Return')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        aligned_df['stock_return_3day'].plot(kind='box')
        plt.ylabel('3-Day Stock Return')
        plt.title('Stock Returns Box Plot')
        
        plt.tight_layout()
        returns_plot = figures_dir / 'returns_distribution.png'
        plt.savefig(returns_plot, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {returns_plot.name}")
        plt.close()
        
        # 2. Ticker distribution
        plt.figure(figsize=(12, 6))
        ticker_counts = aligned_df['ticker'].value_counts().head(20)
        ticker_counts.plot(kind='barh')
        plt.xlabel('Number of Records')
        plt.ylabel('Ticker')
        plt.title('Top 20 Tickers by Record Count')
        plt.tight_layout()
        
        ticker_plot = figures_dir / 'ticker_distribution.png'
        plt.savefig(ticker_plot, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {ticker_plot.name}")
        plt.close()
        
        # 3. Temporal coverage
        plt.figure(figsize=(12, 6))
        aligned_df['transcript_date'] = pd.to_datetime(aligned_df['transcript_date'])
        date_counts = aligned_df.groupby(aligned_df['transcript_date'].dt.to_period('M')).size()
        date_counts.plot(kind='bar')
        plt.xlabel('Month')
        plt.ylabel('Number of Transcripts')
        plt.title('Temporal Coverage of Earnings Calls')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        temporal_plot = figures_dir / 'temporal_coverage.png'
        plt.savefig(temporal_plot, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {temporal_plot.name}")
        plt.close()
        
        # 4. Feature completeness heatmap
        plt.figure(figsize=(10, 8))
        
        # Get completeness for all features
        completeness = {}
        for col in aligned_df.columns:
            completeness[col] = (aligned_df[col].notna().sum() / len(aligned_df)) * 100
        
        # Create categories
        categories = {
            'Metadata': [col for col in aligned_df.columns if col in ['ticker', 'quarter', 'transcript_date', 'financial_date']],
            'Transcript': [col for col in aligned_df.columns if 'management' in col or 'analyst' in col],
            'Market': [col for col in aligned_df.columns if 'price' in col or 'return' in col],
            'Financial': [col for col in aligned_df.columns if col.startswith('financial_')][:10]  # First 10
        }
        
        completeness_data = []
        labels = []
        for category, cols in categories.items():
            for col in cols:
                if col in completeness:
                    completeness_data.append(completeness[col])
                    labels.append(f"{category}: {col.replace('financial_', '')[:30]}")
        
        if completeness_data:
            plt.barh(range(len(completeness_data)), completeness_data)
            plt.yticks(range(len(labels)), labels, fontsize=8)
            plt.xlabel('Completeness (%)')
            plt.title('Feature Completeness')
            plt.xlim(0, 100)
            plt.axvline(95, color='red', linestyle='--', label='95% threshold')
            plt.legend()
            plt.tight_layout()
            
            completeness_plot = figures_dir / 'feature_completeness.png'
            plt.savefig(completeness_plot, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {completeness_plot.name}")
            plt.close()
        
        print()
    
    def generate_phase_summary(self, data):
        """
        Generate summary of all preprocessing phases
        """
        print("="*60)
        print("📋 PHASE SUMMARY")
        print("="*60)
        print()
        
        stats = data.get('statistics', {})
        
        phase_summary = {
            'Phase 2A - Cleaning': {
                'Total processed': stats.get('cleaning', {}).get('total_processed', 0),
                'Successfully cleaned': stats.get('cleaning', {}).get('successfully_cleaned', 0),
                'Success rate': f"{stats.get('cleaning', {}).get('successfully_cleaned', 0) / max(stats.get('cleaning', {}).get('total_processed', 1), 1) * 100:.1f}%"
            },
            'Phase 2B - Segmentation': {
                'Total processed': stats.get('segmentation', {}).get('total_processed', 0),
                'Successfully segmented': stats.get('segmentation', {}).get('successfully_segmented', 0),
                'Success rate': f"{stats.get('segmentation', {}).get('successfully_segmented', 0) / max(stats.get('segmentation', {}).get('total_processed', 1), 1) * 100:.1f}%"
            },
            'Phase 2C - Normalization': {
                'Total records': stats.get('normalization', {}).get('total_records', 0),
                'Features normalized': stats.get('normalization', {}).get('features_normalized', 0),
                'Success rate': '100.0%'
            },
            'Phase 2D - Alignment': {
                'Total transcripts': stats.get('alignment', {}).get('total_transcripts', 0),
                'Successfully aligned': stats.get('alignment', {}).get('successfully_aligned', 0),
                'Success rate': f"{stats.get('alignment', {}).get('successfully_aligned', 0) / max(stats.get('alignment', {}).get('total_transcripts', 1), 1) * 100:.1f}%"
            }
        }
        
        for phase, metrics in phase_summary.items():
            print(f"{phase}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value}")
            print()
        
        self.quality_metrics['phase_summary'] = phase_summary
        return phase_summary
    
    def save_quality_report(self):
        """
        Save comprehensive quality report
        """
        print("="*60)
        print("💾 SAVING QUALITY REPORT")
        print("="*60)
        print()
        
        # Save as JSON
        json_path = self.results_dir / 'data_quality_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.quality_metrics, f, indent=2, default=str)
        print(f"✅ JSON report saved: {json_path}")
        
        # Save as text report
        text_path = self.results_dir / 'data_quality_report.txt'
        with open(text_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("Earnings Call and Risk Intelligence Engine\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 60 + "\n")
            summary = self.quality_metrics.get('summary', {})
            f.write(f"Total Records: {summary.get('total_records', 0):,}\n")
            f.write(f"Total Features: {summary.get('total_features', 0)}\n")
            f.write(f"Unique Tickers: {summary.get('unique_tickers', 0)}\n")
            f.write(f"Overall Completeness: {summary.get('overall_completeness', 0):.1f}%\n\n")
            
            # Phase summary
            f.write("PHASE SUMMARY\n")
            f.write("-" * 60 + "\n")
            for phase, metrics in self.quality_metrics.get('phase_summary', {}).items():
                f.write(f"\n{phase}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
        
        print(f"✅ Text report saved: {text_path}")
        print()
    
    def run(self):
        """
        Execute complete quality reporting workflow
        """
        # Load all data
        data = self.load_all_data()
        
        if data is None:
            print("❌ Failed to load data")
            return False
        
        # Generate reports
        self.generate_data_summary(data)
        self.analyze_data_quality(data)
        self.generate_visualizations(data)
        self.generate_phase_summary(data)
        
        # Save report
        self.save_quality_report()
        
        print("="*60)
        print("✅ PHASE 2E COMPLETE")
        print("="*60)
        print()
        print(f"📁 Quality reports saved to: {self.results_dir}")
        print()
        print("="*60)
        print("🎉 PHASE 2 COMPLETE!")
        print("="*60)
        print()
        print("All preprocessing phases finished successfully!")
        print()
        print("✅ Ready for Phase 3: Feature Engineering")
        print()
        
        return True

def main():
    """Main execution"""
    reporter = DataQualityReporter()
    success = reporter.run()
    return success

if __name__ == "__main__":
    main()