"""
Phase 3A Runner
Executes FinBERT Feature Extraction
"""

from src.features.finbert_extractor import FinBERTExtractor

def run_phase_3a():
    """
    Execute Phase 3A: FinBERT Feature Extraction
    """
    print("\n" + "="*60)
    print("STARTING PHASE 3A: FINBERT FEATURE EXTRACTION")
    print("="*60)
    print()
    print("🤖 This will extract:")
    print("   - 768-dimensional embeddings (mean pooling)")
    print("   - Sentiment scores (negative, neutral, positive)")
    print("   - Management sentiment")
    print("   - Analyst sentiment")
    print("   - Sentiment divergence")
    print()
    print("⏱️  Estimated time: 2-5 minutes for 150 transcripts")
    print()
    
    # Initialize extractor
    extractor = FinBERTExtractor()
    
    # Run extraction
    success = extractor.run()
    
    if not success:
        print("\n❌ Phase 3A failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 3A COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. finbert_features.csv - Full features (768 + 13 dimensions)")
    print("   2. finbert_features_metadata.csv - Sentiment summary")
    print("   3. finbert_extraction_stats.json - Processing statistics")
    print()
    print("📁 Location: data/features/")
    print()
    print("✅ Ready to verify before Phase 3B")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_3a()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 3B:")
        print("1. Run: python verify_phase3a.py")
        print("2. Review FinBERT features")
        print("3. Confirm embeddings extracted")
        print("\nThen commit to Git before Phase 3B")