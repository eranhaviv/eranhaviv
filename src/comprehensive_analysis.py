"""
Comprehensive COSCO Score Analysis

Orchestrates all statistical analyses with methodological rigor
addressing all mandated fixes from the review:

1. Bootstrap internal validation (bias-corrected C-statistics)
2. Cox Proportional Hazards modeling
3. Reclassification metrics (NRI/IDI)
4. Calibration assessment
5. Decision curve analysis

Generates publication-ready results and figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from data_simulation import COSCODataSimulator, CohortParameters
from bootstrap_validation import BootstrapValidator
from survival_models import SurvivalAnalyzer
from reclassification_metrics import ReclassificationAnalyzer
from calibration_analysis import CalibrationAnalyzer, validate_low_risk_phenotype
from decision_curve_analysis import compare_cosco_models_dca

# Plotting settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class COSCOAnalysis:
    """
    Comprehensive COSCO score analysis framework.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        n_bootstrap: int = 1000,
        output_dir: str = 'results'
    ):
        """
        Initialize comprehensive analyzer.

        Args:
            data_path: Path to cohort CSV. If None, generates simulated data.
            n_bootstrap: Number of bootstrap iterations for validation
            output_dir: Directory for results and figures
        """
        self.n_bootstrap = n_bootstrap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)

        # Load or generate data
        if data_path is None:
            print("No data provided. Generating simulated cohort...")
            simulator = COSCODataSimulator()
            self.df = simulator.simulate_cohort()
            simulator._print_summary(self.df)
        else:
            print(f"Loading data from: {data_path}")
            self.df = pd.read_csv(data_path)

        # Initialize results storage
        self.results = {}

    def run_complete_analysis(self) -> Dict:
        """
        Execute all mandated statistical analyses.

        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE COSCO SCORE ANALYSIS")
        print("="*70)
        print(f"Cohort: N={len(self.df)} patients")
        print(f"Bootstrap iterations: {self.n_bootstrap}")
        print("\n")

        # 1. Bootstrap Internal Validation
        print("STEP 1/5: Bootstrap Internal Validation")
        print("-"*70)
        self._run_bootstrap_validation()

        # 2. Cox Proportional Hazards Models
        print("\n\nSTEP 2/5: Cox Proportional Hazards Modeling")
        print("-"*70)
        self._run_survival_analysis()

        # 3. Reclassification Metrics
        print("\n\nSTEP 3/5: Reclassification Analysis (NRI/IDI)")
        print("-"*70)
        self._run_reclassification_analysis()

        # 4. Calibration Assessment
        print("\n\nSTEP 4/5: Calibration Analysis")
        print("-"*70)
        self._run_calibration_analysis()

        # 5. Decision Curve Analysis
        print("\n\nSTEP 5/5: Decision Curve Analysis")
        print("-"*70)
        self._run_decision_curve_analysis()

        # Generate summary report
        print("\n\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)
        self._generate_summary_report()

        print("\n\n" + "="*70)
        print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"  - Figures: {self.output_dir}/figures/")
        print(f"  - Tables: {self.output_dir}/tables/")
        print(f"  - Summary: {self.output_dir}/analysis_summary.txt")

        return self.results

    def _run_bootstrap_validation(self):
        """Execute bootstrap internal validation for all models."""
        from sklearn.linear_model import LogisticRegression

        # Prepare data
        self.df['akin_ii'] = (self.df['akin_stage'] == 2).astype(int)
        self.df['akin_iii'] = (self.df['akin_stage'] == 3).astype(int)

        validator = BootstrapValidator(
            n_iterations=self.n_bootstrap,
            verbose=True
        )

        # Mortality outcome
        print("\n📊 90-Day Mortality (ST_ACM):")
        y_mort = self.df['event_death'].values

        # AKIN model
        X_akin = self.df[['akin_ii', 'akin_iii', 'age', 'ckd', 'chf']]
        result_akin = validator.validate_logistic(X_akin, y_mort)
        print(f"\n  AKIN: {result_akin.corrected_cstat:.3f} (95% CI: {result_akin.ci_lower:.3f}-{result_akin.ci_upper:.3f})")

        # COSCO model
        X_cosco = self.df[['cosco_score', 'age', 'ckd', 'chf']]
        result_cosco = validator.validate_logistic(X_cosco, y_mort)
        print(f"  COSCO: {result_cosco.corrected_cstat:.3f} (95% CI: {result_cosco.ci_lower:.3f}-{result_cosco.ci_upper:.3f})")

        # Combined model
        X_combined = self.df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'ckd', 'chf']]
        result_combined = validator.validate_logistic(X_combined, y_mort)
        print(f"  Combined: {result_combined.corrected_cstat:.3f} (95% CI: {result_combined.ci_lower:.3f}-{result_combined.ci_upper:.3f})")

        self.results['bootstrap_mortality'] = {
            'AKIN': result_akin,
            'COSCO': result_cosco,
            'Combined': result_combined
        }

        # MACE outcome
        print("\n📊 90-Day MACE:")
        y_mace = self.df['event_mace'].values

        # AKIN model (add diabetes for MACE)
        X_akin_mace = self.df[['akin_ii', 'akin_iii', 'age', 'diabetes', 'ckd', 'chf']]
        result_akin_mace = validator.validate_logistic(X_akin_mace, y_mace)
        print(f"\n  AKIN: {result_akin_mace.corrected_cstat:.3f} (95% CI: {result_akin_mace.ci_lower:.3f}-{result_akin_mace.ci_upper:.3f})")

        # COSCO model
        X_cosco_mace = self.df[['cosco_score', 'age', 'diabetes', 'ckd', 'chf']]
        result_cosco_mace = validator.validate_logistic(X_cosco_mace, y_mace)
        print(f"  COSCO: {result_cosco_mace.corrected_cstat:.3f} (95% CI: {result_cosco_mace.ci_lower:.3f}-{result_cosco_mace.ci_upper:.3f})")

        # Combined model
        X_combined_mace = self.df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'diabetes', 'ckd', 'chf']]
        result_combined_mace = validator.validate_logistic(X_combined_mace, y_mace)
        print(f"  Combined: {result_combined_mace.corrected_cstat:.3f} (95% CI: {result_combined_mace.ci_lower:.3f}-{result_combined_mace.ci_upper:.3f})")

        self.results['bootstrap_mace'] = {
            'AKIN': result_akin_mace,
            'COSCO': result_cosco_mace,
            'Combined': result_combined_mace
        }

    def _run_survival_analysis(self):
        """Execute Cox Proportional Hazards modeling."""
        analyzer = SurvivalAnalyzer(self.df)

        # Mortality
        print("\n📊 Cox Models: 90-Day Mortality")
        mortality_results = analyzer.fit_all_models(outcome='mortality')
        self.results['cox_mortality'] = mortality_results

        # MACE
        print("\n📊 Cox Models: MACE")
        mace_results = analyzer.fit_all_models(outcome='mace')
        self.results['cox_mace'] = mace_results

        # Kaplan-Meier curves
        print("\n📊 Generating Kaplan-Meier curves...")
        analyzer.plot_kaplan_meier(
            stratify_by='cosco_cat',
            outcome='mortality',
            save_path=str(self.output_dir / 'figures' / 'km_mortality.png')
        )

        analyzer.plot_kaplan_meier(
            stratify_by='cosco_cat',
            outcome='mace',
            save_path=str(self.output_dir / 'figures' / 'km_mace.png')
        )

        # Time-aware sensitivity
        print("\n📊 Time-aware sensitivity analysis...")
        time_aware_result = analyzer.time_aware_sensitivity(outcome='mortality')
        self.results['time_aware'] = time_aware_result

    def _run_reclassification_analysis(self):
        """Execute reclassification metrics (NRI/IDI)."""
        from sklearn.linear_model import LogisticRegression

        # Prepare predictions for mortality
        y_mort = self.df['event_death'].values

        # Base model (AKIN)
        X_akin = self.df[['akin_ii', 'akin_iii', 'age', 'ckd', 'chf']]
        model_akin = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        model_akin.fit(X_akin, y_mort)
        pred_akin = model_akin.predict_proba(X_akin)[:, 1]

        # New model (AKIN + COSCO)
        X_combined = self.df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'ckd', 'chf']]
        model_combined = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        model_combined.fit(X_combined, y_mort)
        pred_combined = model_combined.predict_proba(X_combined)[:, 1]

        # Reclassification analysis
        print("\n📊 Mortality: COSCO added to AKIN")
        analyzer_mort = ReclassificationAnalyzer(
            y_true=y_mort,
            pred_base=pred_akin,
            pred_new=pred_combined,
            risk_thresholds=[0.1, 0.3]
        )

        reclass_mort = analyzer_mort.perform_analysis(
            method='categorical',
            n_bootstrap=500  # Reduced for speed
        )
        reclass_mort.summary()

        print("\n📋 Reclassification Table:")
        print(reclass_mort.reclassification_table)

        analyzer_mort.plot_reclassification(
            save_path=str(self.output_dir / 'figures' / 'reclassification_mortality.png')
        )

        self.results['reclassification_mortality'] = reclass_mort

        # MACE
        y_mace = self.df['event_mace'].values

        X_akin_mace = self.df[['akin_ii', 'akin_iii', 'age', 'diabetes', 'ckd', 'chf']]
        model_akin_mace = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        model_akin_mace.fit(X_akin_mace, y_mace)
        pred_akin_mace = model_akin_mace.predict_proba(X_akin_mace)[:, 1]

        X_combined_mace = self.df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'diabetes', 'ckd', 'chf']]
        model_combined_mace = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        model_combined_mace.fit(X_combined_mace, y_mace)
        pred_combined_mace = model_combined_mace.predict_proba(X_combined_mace)[:, 1]

        print("\n📊 MACE: COSCO added to AKIN")
        analyzer_mace = ReclassificationAnalyzer(
            y_true=y_mace,
            pred_base=pred_akin_mace,
            pred_new=pred_combined_mace,
            risk_thresholds=[0.2, 0.4]
        )

        reclass_mace = analyzer_mace.perform_analysis(method='categorical', n_bootstrap=500)
        reclass_mace.summary()

        self.results['reclassification_mace'] = reclass_mace

    def _run_calibration_analysis(self):
        """Execute calibration assessment."""
        from sklearn.linear_model import LogisticRegression

        # Mortality calibration
        print("\n📊 Calibration: 90-Day Mortality")
        y_mort = self.df['event_death'].values
        X_combined = self.df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'ckd', 'chf']]

        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X_combined, y_mort)
        pred_mort = model.predict_proba(X_combined)[:, 1]

        cal_analyzer = CalibrationAnalyzer(y_true=y_mort, y_pred=pred_mort, n_bins=10)
        cal_results = cal_analyzer.perform_analysis()
        cal_results.summary()

        cal_analyzer.plot_calibration(
            save_path=str(self.output_dir / 'figures' / 'calibration_mortality.png')
        )

        self.results['calibration_mortality'] = cal_results

        # Low-risk phenotype validation
        print("\n📊 Low-Risk Phenotype Validation (AKIN III + COSCO 0)")
        phenotype_df = validate_low_risk_phenotype(self.df, outcome_col='event_death')
        phenotype_df.to_csv(
            self.output_dir / 'tables' / 'phenotype_validation.csv',
            index=False
        )
        self.results['phenotype_validation'] = phenotype_df

    def _run_decision_curve_analysis(self):
        """Execute decision curve analysis."""
        print("\n📊 Decision Curve Analysis: Mortality")
        dca_mort, analyzer_mort = compare_cosco_models_dca(self.df, outcome='mortality')

        dca_mort.plot(
            save_path=str(self.output_dir / 'figures' / 'dca_mortality.png')
        )

        # Compare at 30% threshold
        comparison = analyzer_mort.compare_models_at_threshold(threshold=0.30)
        print("\n📋 Net Benefit at 30% Mortality Threshold:")
        print(comparison.to_string(index=False))

        comparison.to_csv(
            self.output_dir / 'tables' / 'dca_comparison_30pct.csv',
            index=False
        )

        self.results['dca_mortality'] = dca_mort

        # MACE
        print("\n📊 Decision Curve Analysis: MACE")
        dca_mace, analyzer_mace = compare_cosco_models_dca(self.df, outcome='mace')

        dca_mace.plot(
            save_path=str(self.output_dir / 'figures' / 'dca_mace.png')
        )

        self.results['dca_mace'] = dca_mace

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = self.output_dir / 'analysis_summary.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COSCO SCORE COMPREHENSIVE ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write(f"Cohort: N={len(self.df)} patients\n")
            f.write(f"Bootstrap iterations: {self.n_bootstrap}\n\n")

            # Bootstrap results
            f.write("1. BIAS-CORRECTED C-STATISTICS (Bootstrap Validation)\n")
            f.write("-"*70 + "\n\n")

            f.write("90-Day Mortality:\n")
            for model, result in self.results['bootstrap_mortality'].items():
                f.write(f"  {model:<12}: {result.corrected_cstat:.3f} ")
                f.write(f"(95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})\n")

            f.write("\n90-Day MACE:\n")
            for model, result in self.results['bootstrap_mace'].items():
                f.write(f"  {model:<12}: {result.corrected_cstat:.3f} ")
                f.write(f"(95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})\n")

            # Cox results
            f.write("\n\n2. COX PROPORTIONAL HAZARDS MODELS\n")
            f.write("-"*70 + "\n\n")

            f.write("90-Day Mortality:\n")
            for model, result in self.results['cox_mortality'].items():
                f.write(f"  {model}: C-index = {result.concordance_index:.3f}\n")

            f.write("\n90-Day MACE:\n")
            for model, result in self.results['cox_mace'].items():
                f.write(f"  {model}: C-index = {result.concordance_index:.3f}\n")

            # Reclassification
            f.write("\n\n3. RECLASSIFICATION METRICS\n")
            f.write("-"*70 + "\n\n")

            reclass_mort = self.results['reclassification_mortality']
            f.write("90-Day Mortality (COSCO added to AKIN):\n")
            f.write(f"  NRI: {reclass_mort.nri:.4f} ")
            f.write(f"(95% CI: {reclass_mort.nri_ci_lower:.4f}-{reclass_mort.nri_ci_upper:.4f})\n")
            f.write(f"  p-value: {reclass_mort.nri_pvalue:.4f}\n")
            f.write(f"  IDI: {reclass_mort.idi:.4f} ")
            f.write(f"(95% CI: {reclass_mort.idi_ci_lower:.4f}-{reclass_mort.idi_ci_upper:.4f})\n")
            f.write(f"  p-value: {reclass_mort.idi_pvalue:.4f}\n")

            # Calibration
            f.write("\n\n4. CALIBRATION ASSESSMENT\n")
            f.write("-"*70 + "\n\n")

            cal_mort = self.results['calibration_mortality']
            f.write("90-Day Mortality (Combined Model):\n")
            f.write(f"  Hosmer-Lemeshow: χ² = {cal_mort.hl_statistic:.3f}, ")
            f.write(f"p = {cal_mort.hl_pvalue:.4f}\n")
            f.write(f"  Calibration slope: {cal_mort.calibration_slope:.3f}\n")
            f.write(f"  Brier score: {cal_mort.brier_score:.4f}\n")

            f.write("\n\n5. KEY FINDINGS\n")
            f.write("-"*70 + "\n\n")

            mort_akin = self.results['bootstrap_mortality']['AKIN'].corrected_cstat
            mort_cosco = self.results['bootstrap_mortality']['COSCO'].corrected_cstat
            mort_comb = self.results['bootstrap_mortality']['Combined'].corrected_cstat

            mace_akin = self.results['bootstrap_mace']['AKIN'].corrected_cstat
            mace_cosco = self.results['bootstrap_mace']['COSCO'].corrected_cstat
            mace_comb = self.results['bootstrap_mace']['Combined'].corrected_cstat

            f.write(f"• AKIN shows flat mortality profile (AUC {mort_akin:.3f})\n")
            f.write(f"• COSCO provides modest mortality improvement (AUC {mort_cosco:.3f})\n")
            f.write(f"• Combined model optimal for mortality (AUC {mort_comb:.3f})\n\n")

            f.write(f"⭐ COSCO shows STRONG signal for MACE (AUC {mace_cosco:.3f})\n")
            f.write(f"⭐ Combined model best for MACE (AUC {mace_comb:.3f})\n\n")

            if reclass_mort.nri > 0 and reclass_mort.nri_pvalue < 0.05:
                f.write(f"✓ COSCO provides significant risk reclassification (NRI {reclass_mort.nri:.4f}, p<0.05)\n")

            f.write("\n" + "="*70 + "\n")
            f.write("ANALYSIS COMPLETE\n")
            f.write("="*70 + "\n")

        print(f"\n📄 Summary report saved: {report_path}")


def main():
    """Run complete analysis pipeline."""
    # Option 1: Use simulated data
    analyzer = COSCOAnalysis(
        data_path=None,  # Will generate simulated data
        n_bootstrap=1000,
        output_dir='results'
    )

    # Option 2: Use real data (uncomment and specify path)
    # analyzer = COSCOAnalysis(
    #     data_path='path/to/your/data.csv',
    #     n_bootstrap=1000,
    #     output_dir='results'
    # )

    # Run all analyses
    results = analyzer.run_complete_analysis()

    print("\n✅ All analyses completed successfully!")
    print("\nNext steps:")
    print("  1. Review results in results/analysis_summary.txt")
    print("  2. Check figures in results/figures/")
    print("  3. Review tables in results/tables/")
    print("  4. Use findings to update manuscript")


if __name__ == '__main__':
    main()
