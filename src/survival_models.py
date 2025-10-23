"""
Survival Analysis Module: Cox Proportional Hazards Models

Implements time-to-event modeling for ST_ACM and MACE outcomes,
replacing fixed-time logistic regression with Cox PH regression.

Key Features:
- Cox Proportional Hazards modeling with Hazard Ratios (HRs)
- Proportional hazards assumption testing (Schoenfeld residuals)
- Time-aware sensitivity analysis (complication timing)
- Kaplan-Meier survival curves stratified by COSCO/AKIN
- Cumulative incidence plots

References:
Cox DR. Regression models and life-tables. J R Stat Soc Series B. 1972.
Grambsch PM, Therneau TM. Proportional hazards tests. Biometrika. 1994.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class CoxResults:
    """Results from Cox Proportional Hazards model."""
    model: CoxPHFitter
    concordance_index: float
    hazard_ratios: pd.DataFrame
    log_likelihood: float
    aic: float
    n_events: int
    n_censored: int
    ph_test_pval: float
    ph_assumption_met: bool

    def summary(self):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("COX PROPORTIONAL HAZARDS MODEL RESULTS")
        print("="*70)
        print(f"\nConcordance Index (C-index): {self.concordance_index:.3f}")
        print(f"Log-likelihood: {self.log_likelihood:.2f}")
        print(f"AIC: {self.aic:.2f}")
        print(f"Events: {self.n_events} | Censored: {self.n_censored}")
        print(f"\nProportional Hazards Assumption:")
        print(f"  Global test p-value: {self.ph_test_pval:.4f}")
        print(f"  Assumption met: {'✓ Yes' if self.ph_assumption_met else '✗ No (stratify or time-varying)'}")

        print(f"\n{'Covariate':<20} {'HR':<10} {'95% CI':<20} {'p-value':<10}")
        print("-"*70)
        for idx, row in self.hazard_ratios.iterrows():
            hr = row['HR']
            ci_lower = row['CI_lower']
            ci_upper = row['CI_upper']
            pval = row['p']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

            print(
                f"{idx:<20} "
                f"{hr:<10.3f} "
                f"({ci_lower:.3f}-{ci_upper:.3f})     "
                f"{pval:<10.4f} {sig}"
            )


class SurvivalAnalyzer:
    """
    Comprehensive survival analysis for COSCO score evaluation.
    """

    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize survival analyzer.

        Args:
            df: Cohort dataframe with time-to-event data
            alpha: Significance level for confidence intervals
        """
        self.df = df.copy()
        self.alpha = alpha
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for Cox modeling."""
        # Create AKIN dummy variables
        self.df['akin_ii'] = (self.df['akin_stage'] == 2).astype(int)
        self.df['akin_iii'] = (self.df['akin_stage'] == 3).astype(int)

        # Create COSCO categories (for visualization)
        self.df['cosco_cat'] = pd.cut(
            self.df['cosco_score'],
            bins=[-0.1, 0.5, 1.5, 4.5],
            labels=['COSCO 0', 'COSCO 1', 'COSCO ≥2']
        )

        # Create risk strata for combined model
        # Low: AKIN I-II + COSCO 0
        # Intermediate: AKIN III + COSCO 0, or AKIN I-II + COSCO 1
        # High: COSCO ≥2
        conditions = [
            (self.df['akin_stage'] <= 2) & (self.df['cosco_score'] == 0),
            ((self.df['akin_stage'] == 3) & (self.df['cosco_score'] == 0)) |
            ((self.df['akin_stage'] <= 2) & (self.df['cosco_score'] == 1)),
            self.df['cosco_score'] >= 2
        ]
        self.df['risk_stratum'] = np.select(
            conditions,
            ['Low', 'Intermediate', 'High'],
            default='Intermediate'
        )

    def fit_cox_model(
        self,
        covariates: List[str],
        duration_col: str = 'time_to_death',
        event_col: str = 'event_death',
        penalizer: float = 0.0
    ) -> CoxResults:
        """
        Fit Cox Proportional Hazards model.

        Args:
            covariates: List of covariate column names
            duration_col: Time-to-event column
            event_col: Event indicator (1=event, 0=censored)
            penalizer: L2 penalty for regularization

        Returns:
            CoxResults object
        """
        # Prepare analysis dataset
        analysis_cols = [duration_col, event_col] + covariates
        analysis_df = self.df[analysis_cols].dropna()

        # Fit Cox model
        cph = CoxPHFitter(penalizer=penalizer, alpha=self.alpha)
        cph.fit(
            analysis_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False
        )

        # Extract results
        concordance_index = cph.concordance_index_
        log_likelihood = cph.log_likelihood_
        aic = cph.AIC_

        n_events = analysis_df[event_col].sum()
        n_censored = len(analysis_df) - n_events

        # Hazard ratios with CI
        summary = cph.summary
        hazard_ratios = pd.DataFrame({
            'HR': np.exp(summary['coef']),
            'CI_lower': np.exp(summary['coef lower 95%']),
            'CI_upper': np.exp(summary['coef upper 95%']),
            'p': summary['p']
        })

        # Test proportional hazards assumption
        try:
            ph_test = proportional_hazard_test(
                cph,
                analysis_df,
                time_transform='rank'
            )
            ph_test_pval = ph_test.summary['p'].min()  # Global test (minimum p-value)
            ph_assumption_met = ph_test_pval > self.alpha
        except:
            warnings.warn("Could not perform PH assumption test")
            ph_test_pval = np.nan
            ph_assumption_met = None

        return CoxResults(
            model=cph,
            concordance_index=concordance_index,
            hazard_ratios=hazard_ratios,
            log_likelihood=log_likelihood,
            aic=aic,
            n_events=n_events,
            n_censored=n_censored,
            ph_test_pval=ph_test_pval,
            ph_assumption_met=ph_assumption_met
        )

    def fit_all_models(
        self,
        outcome: str = 'mortality'
    ) -> Dict[str, CoxResults]:
        """
        Fit all three Cox models: AKIN, COSCO, Combined.

        Args:
            outcome: 'mortality' (ST_ACM) or 'mace'

        Returns:
            Dictionary of CoxResults for each model
        """
        if outcome == 'mortality':
            duration_col = 'time_to_death'
            event_col = 'event_death'
        elif outcome == 'mace':
            duration_col = 'time_to_mace'
            event_col = 'event_mace'
        else:
            raise ValueError("outcome must be 'mortality' or 'mace'")

        results = {}

        # Baseline covariates
        baseline = ['age', 'ckd', 'chf']
        if outcome == 'mace':
            baseline.append('diabetes')  # DM important for MACE

        # Model 1: AKIN alone
        print("\n" + "="*70)
        print(f"MODEL 1: AKIN Staging (Outcome: {outcome.upper()})")
        print("="*70)
        akin_vars = ['akin_ii', 'akin_iii'] + baseline
        results['AKIN'] = self.fit_cox_model(akin_vars, duration_col, event_col)
        results['AKIN'].summary()

        # Model 2: COSCO alone
        print("\n" + "="*70)
        print(f"MODEL 2: COSCO Score (Outcome: {outcome.upper()})")
        print("="*70)
        cosco_vars = ['cosco_score'] + baseline
        results['COSCO'] = self.fit_cox_model(cosco_vars, duration_col, event_col)
        results['COSCO'].summary()

        # Model 3: Combined
        print("\n" + "="*70)
        print(f"MODEL 3: AKIN + COSCO Combined (Outcome: {outcome.upper()})")
        print("="*70)
        combined_vars = ['akin_ii', 'akin_iii', 'cosco_score'] + baseline
        results['Combined'] = self.fit_cox_model(combined_vars, duration_col, event_col)
        results['Combined'].summary()

        return results

    def time_aware_sensitivity(
        self,
        outcome: str = 'mortality'
    ) -> CoxResults:
        """
        Sensitivity analysis with time-aware COSCO specification.

        Tests whether early complications (onset ≤3 days) have
        different hazard than late complications.

        Returns:
            CoxResults for time-aware model
        """
        if outcome == 'mortality':
            duration_col = 'time_to_death'
            event_col = 'event_death'
        else:
            duration_col = 'time_to_mace'
            event_col = 'event_mace'

        # Create early/late complication indicators
        self.df['cosco_early'] = (
            (self.df['cosco_score'] > 0) &
            (self.df['complication_onset_time'] <= 3)
        ).astype(int)

        self.df['cosco_late'] = (
            (self.df['cosco_score'] > 0) &
            (self.df['complication_onset_time'] > 3)
        ).astype(int)

        print("\n" + "="*70)
        print(f"TIME-AWARE SENSITIVITY ANALYSIS (Outcome: {outcome.upper()})")
        print("="*70)
        print("Testing: Do early complications (≤3 days) have different hazard?")

        covariates = [
            'akin_ii', 'akin_iii',
            'cosco_early', 'cosco_late',
            'age', 'ckd', 'chf'
        ]
        if outcome == 'mace':
            covariates.append('diabetes')

        result = self.fit_cox_model(covariates, duration_col, event_col)
        result.summary()

        return result

    def plot_kaplan_meier(
        self,
        stratify_by: str = 'cosco_cat',
        outcome: str = 'mortality',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Generate Kaplan-Meier survival curves.

        Args:
            stratify_by: Column to stratify by ('cosco_cat', 'risk_stratum', 'akin_stage')
            outcome: 'mortality' or 'mace'
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if outcome == 'mortality':
            duration_col = 'time_to_death'
            event_col = 'event_death'
            title = '90-Day Survival by {}'
        else:
            duration_col = 'time_to_mace'
            event_col = 'event_mace'
            title = '90-Day MACE-Free Survival by {}'

        title = title.format(stratify_by.replace('_', ' ').title())

        fig, ax = plt.subplots(figsize=figsize)
        kmf = KaplanMeierFitter()

        groups = self.df[stratify_by].unique()
        colors = sns.color_palette("husl", len(groups))

        for group, color in zip(sorted(groups), colors):
            mask = (self.df[stratify_by] == group)
            kmf.fit(
                self.df.loc[mask, duration_col],
                self.df.loc[mask, event_col],
                label=str(group)
            )
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5)

        # Add risk table
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def compare_survival_curves(
        self,
        outcome: str = 'mortality',
        save_path: Optional[str] = None
    ):
        """
        Statistical comparison of survival curves (log-rank test).

        Args:
            outcome: 'mortality' or 'mace'
            save_path: Path to save results table
        """
        if outcome == 'mortality':
            duration_col = 'time_to_death'
            event_col = 'event_death'
        else:
            duration_col = 'time_to_mace'
            event_col = 'event_mace'

        print("\n" + "="*70)
        print(f"LOG-RANK TEST: SURVIVAL CURVE COMPARISONS ({outcome.upper()})")
        print("="*70)

        results = []

        # COSCO categories
        for i, group1 in enumerate(['COSCO 0', 'COSCO 1']):
            for group2 in ['COSCO 1', 'COSCO ≥2']:
                if group1 >= group2:
                    continue

                mask1 = (self.df['cosco_cat'] == group1)
                mask2 = (self.df['cosco_cat'] == group2)

                lr_result = logrank_test(
                    self.df.loc[mask1, duration_col],
                    self.df.loc[mask2, duration_col],
                    self.df.loc[mask1, event_col],
                    self.df.loc[mask2, event_col]
                )

                results.append({
                    'Comparison': f"{group1} vs {group2}",
                    'Chi-square': lr_result.test_statistic,
                    'p-value': lr_result.p_value,
                    'Significant': 'Yes' if lr_result.p_value < 0.05 else 'No'
                })

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"\nSaved: {save_path}")

        return results_df


def main():
    """Example usage."""
    # Load data
    df = pd.read_csv('data/simulated_cohort.csv')

    # Initialize analyzer
    analyzer = SurvivalAnalyzer(df)

    # Fit all models for mortality
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS: 90-DAY MORTALITY (ST_ACM)")
    print("="*70)
    mortality_results = analyzer.fit_all_models(outcome='mortality')

    # Fit all models for MACE
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS: 90-DAY MACE")
    print("="*70)
    mace_results = analyzer.fit_all_models(outcome='mace')

    # Time-aware sensitivity analysis
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: TIME-AWARE COSCO")
    print("="*70)
    time_aware = analyzer.time_aware_sensitivity(outcome='mortality')

    # Kaplan-Meier curves
    print("\n" + "="*70)
    print("KAPLAN-MEIER SURVIVAL CURVES")
    print("="*70)
    analyzer.plot_kaplan_meier(
        stratify_by='cosco_cat',
        outcome='mortality',
        save_path='results/figures/km_curve_mortality.png'
    )

    analyzer.plot_kaplan_meier(
        stratify_by='risk_stratum',
        outcome='mace',
        save_path='results/figures/km_curve_mace.png'
    )

    # Log-rank tests
    analyzer.compare_survival_curves(
        outcome='mortality',
        save_path='results/tables/logrank_tests.csv'
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
