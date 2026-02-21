"""
Phase 3B Runner
Executes Additional NLP Feature Extraction
"""

from src.features.nlp_features import NLPFeatureExtractor

def run_phase_3b():
    """
    Execute Phase 3B: Additional NLP Features
    """
    print("\n" + "="*60)
    print("STARTING PHASE 3B: ADDITIONAL NLP FEATURES")
    print("="*60)
    print()
    print("📚 This will extract:")
    print("   - Loughran-McDonald dictionary features")
    print("   - Text statistics (word count, readability)")
    print("   - Management vs Analyst divergence metrics")
    print()
    print("⏱️  Estimated time: 2-3 minutes for 1,288 transcripts")
    print()
    
    # Initialize extractor
    extractor = NLPFeatureExtractor()
    
    # Run extraction
    success = extractor.run()
    
    if not success:
        print("\n❌ Phase 3B failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 3B COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. nlp_features.csv - NLP features")
    print("   2. nlp_extraction_stats.json - Statistics")
    print()
    print("📁 Location: data/features/")
    print()
    print("✅ Ready to verify before Phase 3C")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_3b()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 3C:")
        print("1. Run: python verify_phase3b.py")
        print("2. Review NLP features")
        print("3. Confirm extraction success")
        print("\nThen commit to Git before Phase 3C")