"""
Phase 2A Runner
Executes Transcript Cleaning and Normalization
"""

from src.preprocessing.transcript_cleaner import TranscriptCleaner

def run_phase_2a():
    """
    Execute Phase 2A: Transcript Cleaning
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2A: TRANSCRIPT CLEANING")
    print("="*60)
    print()
    
    # Initialize cleaner
    cleaner = TranscriptCleaner()
    
    # Run cleaning process
    success = cleaner.run()
    
    if not success:
        print("\n❌ Phase 2A failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 2A COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. transcripts_cleaned.json - Full cleaned transcripts")
    print("   2. transcripts_cleaned_metadata.csv - Metadata summary")
    print("   3. cleaning_statistics.json - Processing statistics")
    print()
    print("📁 Location: data/processed/transcripts/")
    print()
    print("✅ Ready to verify before Phase 2B")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_2a()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 2B:")
        print("1. Run: python verify_phase2a.py")
        print("2. Review cleaning statistics")
        print("3. Confirm quality metrics")
        print("\nThen commit to Git before Phase 2B")