"""
Phase 6 Runner
Executes Baseline Model Training
6A: Logistic Regression (financial-only)
6B: Logistic Regression (sentiment-only)
6C: Combined model + evaluation metrics
6D: Baseline performance documentation
"""

from src.models.baseline_models import BaselineModels

def run_phase_6():
    print("\n" + "="*60)
    print("STARTING PHASE 6: BASELINE MODELS")
    print("="*60)
    print()
    print("🤖 This will train:")
    print("   6A: Logistic Regression + Random Forest (financial features only)")
    print("   6B: Logistic Regression + Random Forest (sentiment features only)")
    print("   6C: Combined model (financial + sentiment + market)")
    print("   6D: Comparison table + performance documentation")
    print()
    print("⏱️  Estimated time: 5-10 minutes")
    print()

    trainer = BaselineModels()
    trainer.run()

    print("✅ Ready for Phase 7: XGBoost Models")
    print()
    return True

if __name__ == "__main__":
    run_phase_6()