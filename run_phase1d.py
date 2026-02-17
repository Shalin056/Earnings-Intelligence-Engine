"""
Phase 1D Runner
Executes Earnings Call Transcript Collection
"""

from src.data_collection.transcript_scraper import TranscriptCollector
from config import RAW_DATA_DIR

def run_phase_1d():
    """
    Execute Phase 1D
    """
    print("\n" + "="*60)
    print("STARTING PHASE 1D")
    print("="*60)
    print()
    
    # Collect transcripts
    collector = TranscriptCollector(target_count=150)
    transcripts = collector.run()
    
    if transcripts is None or len(transcripts) == 0:
        print("\n❌ Phase 1D failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 1D COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print(f"   1. Transcript metadata: {len(transcripts)} records")
    print(f"   2. Full transcript JSON")
    print(f"   3. Individual transcript files")
    print()
    print("📁 Data location:", RAW_DATA_DIR / 'transcripts')
    print()
    print("✅ Ready for Phase 2: Data Preprocessing")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_1d()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Waiting for your confirmation")
        print("="*60)
        print("\nPlease verify:")
        print("1. Check the data/raw/transcripts/ folder")
        print("2. Review sample transcripts")
        print("3. Decide on manual collection strategy")
        print("\nWhen ready, confirm to proceed to Phase 2")