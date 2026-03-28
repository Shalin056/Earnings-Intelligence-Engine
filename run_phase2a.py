"""
Phase 2A Runner
Executes Transcript Cleaning
"""

from src.preprocessing.transcript_cleaner import TranscriptCleaner

def run_phase_2a(real_only=False):
    """
    Execute Phase 2A: Transcript Cleaning
    
    Args:
        real_only: If True, only process real transcripts (for testing)
                  If False, process all transcripts (for production)
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2A: TRANSCRIPT CLEANING")
    print("="*60)
    print()
    
    if real_only:
        print("⚠️  REAL-ONLY MODE ENABLED")
        print("   Only real transcripts will be processed")
        print("   (Set real_only=False for full dataset)")
        print()
    
    cleaner = TranscriptCleaner()
    success = cleaner.run(real_only=real_only)
    
    if success:
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
    
    return success


if __name__ == "__main__":
    # FOR PRODUCTION: Use all transcripts
    # run_phase_2a(real_only=False)
    
    # FOR TESTING REAL ONLY: Uncomment below
    run_phase_2a(real_only=True)