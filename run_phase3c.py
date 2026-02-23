"""
Phase 3C Runner
Executes Feature Integration
"""

from src.features.feature_integrator import FeatureIntegrator

def run_phase_3c():
    """
    Execute Phase 3C: Feature Integration
    """
    print("\n" + "="*60)
    print("STARTING PHASE 3C: FEATURE INTEGRATION")
    print("="*60)
    print()
    print("🔗 This will:")
    print("   - Merge FinBERT + NLP + Financial + Market features")
    print("   - Create final training dataset")
    print("   - Split into train/test sets (80/20)")
    print("   - Prepare for modeling")
    print()
    print("⏱️  Estimated time: 1-2 minutes")
    print()
    
    # Initialize integrator
    integrator = FeatureIntegrator()
    
    # Run integration
    success = integrator.run()
    
    if not success:
        print("\n❌ Phase 3C failed.")
        return False
    
    # Final summary
    print("="*60)
    print("🎉 PHASE 3C COMPLETE!")
    print("="*60)
    print()
    print("✅ Deliverables:")
    print("   1. final_dataset.csv - Complete dataset")
    print("   2. train_data.csv - Training set (70%)")
    print("   3. test_data.csv - Test set (30%)")
    print("   4. feature_info.json - Feature metadata")
    print("   5. integration_stats.json - Integration statistics")
    print()
    print("📁 Location: data/final/")
    print()
    print("="*60)
    print("🎊 PHASE 3 (FEATURE ENGINEERING) COMPLETE!")
    print("="*60)
    print()
    print("Next: Phase 4 - Baseline Modeling")
    print()
    
    return True

if __name__ == "__main__":
    run_phase_3c()