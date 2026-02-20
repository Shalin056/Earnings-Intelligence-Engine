"""
Phase 2D Runner
Executes Temporal Alignment
"""

from src.preprocessing.temporal_aligner import TemporalAligner

def run_phase_2d():
    """
    Execute Phase 2D: Temporal Alignment
    """
    print("\n" + "="*60)
    print("STARTING PHASE 2D: TEMPORAL ALIGNMENT")
    print("="*60)
    print()
    print("⚠️  CRITICAL PHASE: Preventing Lookahead Bias")
    print()
    
    # Initialize aligner
    aligner = TemporalAligner()
    
    # Run alignment process
    success = aligner.run()
    
    if not success:
        print("\n❌ Phase 2D failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 2D COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. aligned_data.csv - Temporally aligned dataset")
    print("   2. alignment_statistics.json - Alignment metrics")
    print("   3. alignment_metadata.json - Dataset information")
    print()
    print("📁 Location: data/processed/aligned/")
    print()
    print("⚠️  IMPORTANT: Temporal integrity validated - No lookahead bias!")
    print()
    print("✅ Ready to verify before Phase 2E")
    print()
    
    return True

if __name__ == "__main__":
    success = run_phase_2d()
    
    if success:
        print("="*60)
        print("⏸️  PAUSED - Verification Required")
        print("="*60)
        print("\nBefore proceeding to Phase 2E:")
        print("1. Run: python verify_phase2d.py")
        print("2. Review temporal alignment")
        print("3. Confirm no lookahead bias")
        print("\nThen commit to Git before Phase 2E")