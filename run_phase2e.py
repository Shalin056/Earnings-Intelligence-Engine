"""
Phase 2E Runner
Executes Data Quality Reporting
"""

from src.preprocessing.quality_reporter import DataQualityReporter

def run_phase_2e():
    """
    Execute Phase 2E: Data Quality Reporting
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2E: DATA QUALITY REPORTING")
    print("="*60)
    print()
    
    # Initialize reporter
    reporter = DataQualityReporter()
    
    # Run reporting process
    success = reporter.run()
    
    if not success:
        print("\n❌ Phase 2E failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 2E COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. data_quality_report.json - Comprehensive metrics")
    print("   2. data_quality_report.txt - Human-readable report")
    print("   3. Visualization plots:")
    print("      - returns_distribution.png")
    print("      - ticker_distribution.png")
    print("      - temporal_coverage.png")
    print("      - feature_completeness.png")
    print()
    print("📁 Location: results/quality_reports/")
    print()
    print("="*60)
    print("🎊 PHASE 2 (PREPROCESSING) COMPLETE!")
    print("="*60)
    print()
    print("Next: Phase 3 - Feature Engineering")
    print()
    
    return True

if __name__ == "__main__":
    run_phase_2e()