"""
Phase 7 Runner
Executes XGBoost Model Training
"""

from src.models.xgboost_models import XGBoostModels

def run_phase_7():
    """
    Execute Phase 7: XGBoost Models
    """
    print("\n" + "="*60)
    print("STARTING PHASE 7: XGBOOST MODELS")
    print("="*60)
    print()
    print("🚀 This will train:")
    print("   7A: XGBoost with financial features only")
    print("   7B: XGBoost with sentiment features only")
    print("   7C: XGBoost with combined features (no embeddings)")
    print("   7D: XGBoost with FULL features (including FinBERT)")
    print()
    print("Expected improvements:")
    print("   Baseline (Phase 6): 57.9% ROC-AUC")
    print("   Target (Phase 7):   59-62% ROC-AUC")
    print()
    print("⏱️  Estimated time: 5-10 minutes")
    print()
    
    trainer = XGBoostModels()
    results = trainer.run()
    
    print("✅ Ready for Phase 8: Model Comparison & Visualization")
    print()
    
    return results


if __name__ == "__main__":
    run_phase_7()