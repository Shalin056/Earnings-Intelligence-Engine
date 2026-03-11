"""
Phase 8 Runner
Executes Model Validation
8A: Time-series cross-validation
8B: SHAP interpretability
8C: Error analysis
8D: Statistical testing (DeLong)
"""

from src.models.model_validator import ModelValidator

def run_phase_8():
    """
    Execute Phase 8: Model Validation
    """
    print("\n" + "="*60)
    print("STARTING PHASE 8: MODEL VALIDATION")
    print("="*60)
    print()
    print("📊 This will perform:")
    print("   8A: 5-fold time-series cross-validation")
    print("   8B: SHAP interpretability analysis")
    print("   8C: Error analysis (FP/FN breakdown)")
    print("   8D: Statistical significance testing (DeLong test)")
    print()
    print("⏱️  Estimated time: 10-15 minutes")
    print()
    
    validator = ModelValidator()
    success = validator.run()
    
    if success:
        print("="*60)
        print("🎉 PHASE 8 COMPLETE!")
        print("="*60)
        print()
        print("✅ Deliverables:")
        print("   1. phase8a_cross_validation_results.json")
        print("   2. phase8b_feature_importance.csv")
        print("   3. phase8c_error_analysis.csv")
        print("   4. phase8d_statistical_tests.csv")
        print("   5. phase8_validation_report.json")
        print("   6. shap_summary_plot.png")
        print("   7. error_analysis.png")
        print()
        print("📁 Location: results/validation/")
        print()
        print("✅ Models validated and ready for deployment!")
        print()
    
    return success


if __name__ == "__main__":
    run_phase_8()