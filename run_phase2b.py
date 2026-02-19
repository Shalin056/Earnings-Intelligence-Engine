"""
Phase 2B Runner
Executes Speaker Segmentation
"""

from src.preprocessing.speaker_segmenter import SpeakerSegmenter

def run_phase_2b():
    """
    Execute Phase 2B: Speaker Segmentation
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2B: SPEAKER SEGMENTATION")
    print("="*60)
    print()
    
    # Initialize segmenter
    segmenter = SpeakerSegmenter()
    
    # Run segmentation process
    success = segmenter.run()
    
    if not success:
        print("\n❌ Phase 2B failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 2B COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. transcripts_segmented.json - Segmented by speaker role")
    print("   2. transcripts_segmented_metadata.csv - Segmentation stats")
    print("   3. segmentation_statistics.json - Processing statistics")
    print()
    print("📁 Location: data/processed/transcripts/")
    print()
    print("✅ Ready to verify before Phase 2C")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_2b()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 2C:")
        print("1. Run: python verify_phase2b.py")
        print("2. Review segmentation statistics")
        print("3. Confirm quality metrics")
        print("\nThen commit to Git before Phase 2C")