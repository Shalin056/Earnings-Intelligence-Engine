"""
Phase 2C Runner
Executes Financial Data Normalization
"""

from src.preprocessing.financial_normalizer import FinancialNormalizer

def run_phase_2c():
    """
    Execute Phase 2C: Financial Data Normalization
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2C: FINANCIAL DATA NORMALIZATION")
    print("="*60)
    print()
    
    # Initialize normalizer
    normalizer = FinancialNormalizer()
    
    # Run normalization process
    success = normalizer.run()
    
    if not success:
        print("\n❌ Phase 2C failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 2C COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. financial_features_normalized.csv - Normalized financial data")
    print("   2. robust_scaler.pkl - Fitted scaler for future use")
    print("   3. normalization_metadata.json - Feature information")
    print("   4. normalization_statistics.json - Processing statistics")
    print()
    print("📁 Location: data/processed/financial/")
    print()
    print("✅ Ready to verify before Phase 2D")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_2c()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 2D:")
        print("1. Run: python verify_phase2c.py")
        print("2. Review normalization statistics")
        print("3. Confirm all features normalized")
        print("\nThen commit to Git before Phase 2D")