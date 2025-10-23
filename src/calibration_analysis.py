"""
Calibration Analysis Module

Implements calibration assessment for prognostic models,
evaluating how well predicted probabilities match observed outcomes.

Key Features:
- Calibration plots (observed vs. predicted)
- Hosmer-Lemeshow goodness-of-fit test
- Calibration slope and intercept
- Calibration-in-the-large
- Flexible binning strategies (deciles, fixed bins)

Critical for validating low-risk phenotype (AKIN III + COSCO 0).

References:
Hosmer DW, Lemeshow S. Applied Logistic Regression. 2000.
Austin PC, Steyerberg EW. Events per variable in logistic regression.
Stat Med. 2017.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CalibrationResults:
    """Results from calibration analysis."""
    hl_statistic: float
    hl_pvalue: float
    hl_df: int
    calibration_slope: float
    calibration_intercept: float
    calibration_in_large: float
    brier_score: float
    calibration_bins: pd.DataFrame

    def summary(self):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("CALIBRATION ANALYSIS RESULTS")
        print("="*70)

        print(f"\nHosmer-Lemeshow Goodness-of-Fit Test:")
        print(f"  Chi-square statistic: {self.hl_statistic:.3f}")
        print(f"  Degrees of freedom: {self.hl_df}")
        print(f"  p-value: {self.hl_pvalue:.4f}")

        if self.hl_pvalue > 0.05:
            print(f"  Interpretation: ✓ Good calibration (p > 0.05, fail to reject)")
        else:
            print(f"  Interpretation: ✗ Poor calibration (p < 0.05, reject null)")

        print(f"\nCalibration Metrics:")
        print(f"  Calibration slope: {self.calibration_slope:.3f}")
        if abs(self.calibration_slope - 1.0) < 0.1:
            print(f"    ✓ Excellent (close to 1.0)")
        else:
            print(f"    ⚠ {self.calibration_slope:.3f} ≠ 1.0 (recalibration may be needed)")

        print(f"  Calibration intercept: {self.calibration_intercept:.3f}")
        if abs(self.calibration_intercept) < 0.1:
            print(f"    ✓ Excellent (close to 0.0)")
        else:
            print(f"    ⚠ {self.calibration_intercept:.3f} ≠ 0.0 (systematic over/under-prediction)")

        print(f"  Calibration-in-the-large: {self.calibration_in_large:.3f}")

        print(f"\nBrier Score: {self.brier_score:.4f}")
        print(f"  (Lower is better; maximum = 0.25 for random classifier)")


class CalibrationAnalyzer:
    """
    Calibration assessment for prognostic models.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ):
        """
        Initialize calibration analyzer.

        Args:
            y_true: True binary outcomes (0/1)
            y_pred: Predicted probabilities (0-1)
            n_bins: Number of bins for calibration plot
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.n_bins = n_bins
        self.n = len(y_true)

    def hosmer_lemeshow_test(self) -> Tuple[float, float, int]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.

        Algorithm:
        1. Bin patients by predicted risk (typically deciles)
        2. For each bin, calculate:
           - Observed events (O)
           - Expected events (E) = sum of predicted probabilities
        3. Chi-square statistic: Σ[(O - E)² / E]
        4. Compare to chi-square distribution with (n_bins - 2) df

        Returns:
            (chi_square_statistic, p_value, degrees_of_freedom)
        """
        # Create bins based on predicted probabilities
        bin_edges = np.percentile(
            self.y_pred,
            np.linspace(0, 100, self.n_bins + 1)
        )
        bin_edges[-1] += 1e-6  # Ensure last value included
        bins = np.digitize(self.y_pred, bin_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)

        # Calculate observed and expected for each bin
        chi_square = 0.0

        for i in range(self.n_bins):
            bin_mask = (bins == i)
            n_bin = bin_mask.sum()

            if n_bin == 0:
                continue

            # Observed events
            observed = self.y_true[bin_mask].sum()

            # Expected events
            expected = self.y_pred[bin_mask].sum()

            # Observed non-events
            observed_nonevent = n_bin - observed

            # Expected non-events
            expected_nonevent = n_bin - expected

            # Chi-square contribution (events and non-events)
            if expected > 0:
                chi_square += (observed - expected) ** 2 / expected

            if expected_nonevent > 0:
                chi_square += (observed_nonevent - expected_nonevent) ** 2 / expected_nonevent

        # Degrees of freedom
        df = self.n_bins - 2

        # p-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        return chi_square, p_value, df

    def calibration_slope_intercept(self) -> Tuple[float, float]:
        """
        Calculate calibration slope and intercept.

        Method:
        Fit logistic regression: logit(y) = α + β × logit(predicted_prob)

        Ideal calibration:
            α (intercept) = 0
            β (slope) = 1

        Returns:
            (calibration_slope, calibration_intercept)
        """
        # Convert predicted probabilities to logit scale
        # Add small epsilon to avoid log(0) or log(1)
        epsilon = 1e-7
        y_pred_clipped = np.clip(self.y_pred, epsilon, 1 - epsilon)
        logit_pred = np.log(y_pred_clipped / (1 - y_pred_clipped))

        # Fit logistic regression
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(logit_pred.reshape(-1, 1), self.y_true)

        calibration_slope = model.coef_[0][0]
        calibration_intercept = model.intercept_[0]

        return calibration_slope, calibration_intercept

    def calibration_in_large(self) -> float:
        """
        Calculate calibration-in-the-large.

        Measures systematic over or under-prediction:
            mean(observed) - mean(predicted)

        Ideal: 0.0
        """
        observed_mean = self.y_true.mean()
        predicted_mean = self.y_pred.mean()

        return observed_mean - predicted_mean

    def create_calibration_bins(self) -> pd.DataFrame:
        """
        Create calibration table with bins.

        Returns:
            DataFrame with columns:
                - bin: Bin number
                - n: Number of patients
                - predicted_mean: Mean predicted probability
                - observed_rate: Observed event rate
                - ci_lower, ci_upper: 95% CI for observed rate
        """
        # Create bins
        bin_edges = np.percentile(
            self.y_pred,
            np.linspace(0, 100, self.n_bins + 1)
        )
        bin_edges[-1] += 1e-6
        bins = np.digitize(self.y_pred, bin_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)

        calibration_data = []

        for i in range(self.n_bins):
            bin_mask = (bins == i)
            n_bin = bin_mask.sum()

            if n_bin == 0:
                continue

            # Predicted mean
            predicted_mean = self.y_pred[bin_mask].mean()

            # Observed rate
            observed_rate = self.y_true[bin_mask].mean()

            # 95% CI for observed rate (Wilson score interval)
            n_events = self.y_true[bin_mask].sum()
            ci_lower, ci_upper = self._wilson_ci(n_events, n_bin)

            calibration_data.append({
                'bin': i + 1,
                'n': n_bin,
                'predicted_mean': predicted_mean,
                'observed_rate': observed_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        return pd.DataFrame(calibration_data)

    def _wilson_ci(
        self,
        n_events: int,
        n_total: int,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for proportion.

        More accurate than normal approximation for small samples.
        """
        if n_total == 0:
            return 0.0, 0.0

        p = n_events / n_total
        z = stats.norm.ppf(1 - alpha / 2)

        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt((p * (1 - p) / n_total) + (z**2 / (4 * n_total**2))) / denominator

        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)

        return ci_lower, ci_upper

    def perform_analysis(self) -> CalibrationResults:
        """
        Perform complete calibration analysis.

        Returns:
            CalibrationResults object
        """
        # Hosmer-Lemeshow test
        hl_stat, hl_pval, hl_df = self.hosmer_lemeshow_test()

        # Calibration slope and intercept
        cal_slope, cal_intercept = self.calibration_slope_intercept()

        # Calibration-in-the-large
        cal_in_large = self.calibration_in_large()

        # Brier score
        brier = brier_score_loss(self.y_true, self.y_pred)

        # Calibration bins
        cal_bins = self.create_calibration_bins()

        return CalibrationResults(
            hl_statistic=hl_stat,
            hl_pvalue=hl_pval,
            hl_df=hl_df,
            calibration_slope=cal_slope,
            calibration_intercept=cal_intercept,
            calibration_in_large=cal_in_large,
            brier_score=brier,
            calibration_bins=cal_bins
        )

    def plot_calibration(
        self,
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ):
        """
        Create calibration plot (observed vs. predicted).

        Perfect calibration: points lie on 45-degree diagonal.
        """
        cal_bins = self.create_calibration_bins()

        fig, ax = plt.subplots(figsize=figsize)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect calibration')

        # Calibration curve with error bars
        ax.errorbar(
            cal_bins['predicted_mean'],
            cal_bins['observed_rate'],
            yerr=[
                cal_bins['observed_rate'] - cal_bins['ci_lower'],
                cal_bins['ci_upper'] - cal_bins['observed_rate']
            ],
            fmt='o-',
            linewidth=2,
            markersize=8,
            capsize=5,
            color='steelblue',
            label='Observed'
        )

        # Add histogram of predicted probabilities (rug plot)
        ax2 = ax.twinx()
        ax2.hist(
            self.y_pred,
            bins=30,
            alpha=0.2,
            color='gray',
            label='Distribution'
        )
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.tick_params(axis='y', labelsize=9)

        # Styling
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observed Event Rate', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()


def validate_low_risk_phenotype(
    df: pd.DataFrame,
    outcome_col: str = 'event_death'
) -> pd.DataFrame:
    """
    Special validation for AKIN III + COSCO 0 low-risk phenotype.

    This is a critical clinical finding that requires careful validation
    with proper confidence intervals due to small cell counts.

    Args:
        df: Cohort dataframe
        outcome_col: Outcome column name

    Returns:
        DataFrame with mortality rates by AKIN/COSCO strata
    """
    print("\n" + "="*70)
    print("LOW-RISK PHENOTYPE VALIDATION: AKIN III + COSCO 0")
    print("="*70)

    # Create AKIN × COSCO strata
    strata_results = []

    for akin in [1, 2, 3]:
        for cosco in [0, 1, 2, 3, 4]:
            mask = (df['akin_stage'] == akin) & (df['cosco_score'] == cosco)
            n = mask.sum()

            if n == 0:
                continue

            # Observed outcomes
            n_events = df.loc[mask, outcome_col].sum()
            event_rate = n_events / n

            # Wilson 95% CI
            analyzer = CalibrationAnalyzer(
                y_true=df.loc[mask, outcome_col].values,
                y_pred=np.full(n, event_rate)  # Placeholder
            )
            ci_lower, ci_upper = analyzer._wilson_ci(n_events, n)

            strata_results.append({
                'AKIN': akin,
                'COSCO': cosco,
                'N': n,
                'Events': int(n_events),
                'Rate': event_rate,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper
            })

    results_df = pd.DataFrame(strata_results)

    # Highlight AKIN III + COSCO 0
    print("\nAKIN × COSCO Mortality Rates:")
    print(results_df.to_string(index=False))

    akin3_cosco0 = results_df[
        (results_df['AKIN'] == 3) & (results_df['COSCO'] == 0)
    ]

    if len(akin3_cosco0) > 0:
        row = akin3_cosco0.iloc[0]
        print("\n" + "="*70)
        print("⭐ LOW-RISK PHENOTYPE: AKIN III + COSCO 0")
        print("="*70)
        print(f"N: {row['N']}")
        print(f"Events: {row['Events']}")
        print(f"Mortality Rate: {row['Rate']*100:.2f}%")
        print(f"95% CI: ({row['CI_lower']*100:.2f}% - {row['CI_upper']*100:.2f}%)")

        if row['Rate'] < 0.05:
            print("\n✓ Confirms very low mortality despite severe AKIN stage")
            print("  Clinical interpretation: Isolated renal injury without systemic fallout")
        else:
            print("\n⚠ Mortality not as low as expected; external validation needed")

    return results_df


def main():
    """Example usage."""
    # Load data
    df = pd.read_csv('data/simulated_cohort.csv')

    # Fit a simple logistic model to get predictions
    print("="*70)
    print("CALIBRATION ANALYSIS: COSCO SCORE MODEL")
    print("="*70)

    from sklearn.linear_model import LogisticRegression

    X = df[['cosco_score', 'age', 'ckd', 'chf']]
    y = df['event_death'].values

    model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]

    # Calibration analysis
    analyzer = CalibrationAnalyzer(y_true=y, y_pred=y_pred, n_bins=10)
    results = analyzer.perform_analysis()
    results.summary()

    # Calibration plot
    analyzer.plot_calibration(
        save_path='results/figures/calibration_plot.png'
    )

    # Low-risk phenotype validation
    phenotype_results = validate_low_risk_phenotype(df, outcome_col='event_death')


if __name__ == '__main__':
    main()
