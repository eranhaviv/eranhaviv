"""
Reclassification Metrics Module

Implements Net Reclassification Improvement (NRI) and
Integrated Discrimination Improvement (IDI) to quantify
the clinical added value of COSCO score beyond AKIN staging.

Key Features:
- Categorical NRI with clinically meaningful risk thresholds
- Continuous NRI (category-free)
- IDI calculation
- Bootstrap confidence intervals for both metrics
- Reclassification tables
- Visualization of risk reclassification

References:
Pencina MJ, D'Agostino RB, et al. Evaluating the added predictive ability
of a new marker. Stat Med. 2008.
Cook NR. Use and misuse of the receiver operating characteristic curve.
Circulation. 2007.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from sklearn.metrics import brier_score_loss


@dataclass
class ReclassificationResults:
    """Results from reclassification analysis."""
    nri: float
    nri_events: float
    nri_nonevents: float
    nri_pvalue: float
    nri_ci_lower: float
    nri_ci_upper: float
    idi: float
    idi_pvalue: float
    idi_ci_lower: float
    idi_ci_upper: float
    reclassification_table: pd.DataFrame

    def summary(self):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("RECLASSIFICATION ANALYSIS RESULTS")
        print("="*70)

        print(f"\nNet Reclassification Improvement (NRI):")
        print(f"  Overall NRI: {self.nri:.4f} (95% CI: {self.nri_ci_lower:.4f} to {self.nri_ci_upper:.4f})")
        print(f"  p-value: {self.nri_pvalue:.4f}")
        sig = "***" if self.nri_pvalue < 0.001 else "**" if self.nri_pvalue < 0.01 else "*" if self.nri_pvalue < 0.05 else "ns"
        print(f"  Significance: {sig}")

        print(f"\n  Components:")
        print(f"    NRI (Events): {self.nri_events:.4f} (correct upward reclassification)")
        print(f"    NRI (Non-events): {self.nri_nonevents:.4f} (correct downward reclassification)")

        print(f"\nIntegrated Discrimination Improvement (IDI):")
        print(f"  IDI: {self.idi:.4f} (95% CI: {self.idi_ci_lower:.4f} to {self.idi_ci_upper:.4f})")
        print(f"  p-value: {self.idi_pvalue:.4f}")
        sig = "***" if self.idi_pvalue < 0.001 else "**" if self.idi_pvalue < 0.01 else "*" if self.idi_pvalue < 0.05 else "ns"
        print(f"  Significance: {sig}")

        print(f"\nInterpretation:")
        if self.nri > 0 and self.nri_pvalue < 0.05:
            print("  ✓ COSCO provides significant risk reclassification benefit")
            print(f"    {self.nri*100:.1f}% net improvement in risk classification")
        else:
            print("  ✗ No significant reclassification benefit demonstrated")

        if self.idi > 0 and self.idi_pvalue < 0.05:
            print("  ✓ COSCO improves average predicted risk separation")
        else:
            print("  ✗ No significant improvement in risk discrimination")


class ReclassificationAnalyzer:
    """
    Calculate reclassification metrics for prognostic models.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        pred_base: np.ndarray,
        pred_new: np.ndarray,
        risk_thresholds: Optional[List[float]] = None
    ):
        """
        Initialize reclassification analyzer.

        Args:
            y_true: True binary outcomes (0/1)
            pred_base: Predicted probabilities from base model (AKIN)
            pred_new: Predicted probabilities from new model (AKIN + COSCO)
            risk_thresholds: List of risk category boundaries
                           Default: [0.1, 0.3] → Low (<10%), Intermediate (10-30%), High (>30%)
        """
        self.y_true = np.array(y_true)
        self.pred_base = np.array(pred_base)
        self.pred_new = np.array(pred_new)

        if risk_thresholds is None:
            self.risk_thresholds = [0.1, 0.3]
        else:
            self.risk_thresholds = sorted(risk_thresholds)

        self.n = len(y_true)
        self.n_events = self.y_true.sum()
        self.n_nonevents = self.n - self.n_events

    def calculate_nri(
        self,
        method: str = 'categorical'
    ) -> Tuple[float, float, float]:
        """
        Calculate Net Reclassification Improvement.

        Args:
            method: 'categorical' (risk categories) or 'continuous' (category-free)

        Returns:
            (NRI_total, NRI_events, NRI_nonevents)
        """
        if method == 'categorical':
            return self._categorical_nri()
        elif method == 'continuous':
            return self._continuous_nri()
        else:
            raise ValueError("method must be 'categorical' or 'continuous'")

    def _categorical_nri(self) -> Tuple[float, float, float]:
        """
        Categorical NRI using predefined risk thresholds.

        Algorithm:
        1. Assign base and new predictions to risk categories
        2. For events: P(move up) - P(move down)
        3. For non-events: P(move down) - P(move up)
        4. NRI = NRI_events + NRI_nonevents
        """
        # Create risk categories
        cat_base = self._categorize_risk(self.pred_base)
        cat_new = self._categorize_risk(self.pred_new)

        # Calculate movement for events
        events_mask = (self.y_true == 1)
        move_up_events = np.sum((cat_new > cat_base) & events_mask)
        move_down_events = np.sum((cat_new < cat_base) & events_mask)

        nri_events = (move_up_events - move_down_events) / self.n_events

        # Calculate movement for non-events
        nonevents_mask = (self.y_true == 0)
        move_up_nonevents = np.sum((cat_new > cat_base) & nonevents_mask)
        move_down_nonevents = np.sum((cat_new < cat_base) & nonevents_mask)

        nri_nonevents = (move_down_nonevents - move_up_nonevents) / self.n_nonevents

        # Total NRI
        nri = nri_events + nri_nonevents

        return nri, nri_events, nri_nonevents

    def _continuous_nri(self) -> Tuple[float, float, float]:
        """
        Category-free (continuous) NRI.

        For events: proportion with increased predicted risk
        For non-events: proportion with decreased predicted risk
        """
        # For events
        events_mask = (self.y_true == 1)
        move_up_events = np.sum((self.pred_new > self.pred_base) & events_mask)
        move_down_events = np.sum((self.pred_new < self.pred_base) & events_mask)

        nri_events = (move_up_events - move_down_events) / self.n_events

        # For non-events
        nonevents_mask = (self.y_true == 0)
        move_up_nonevents = np.sum((self.pred_new > self.pred_base) & nonevents_mask)
        move_down_nonevents = np.sum((self.pred_new < self.pred_base) & nonevents_mask)

        nri_nonevents = (move_down_nonevents - move_up_nonevents) / self.n_nonevents

        # Total NRI
        nri = nri_events + nri_nonevents

        return nri, nri_events, nri_nonevents

    def _categorize_risk(self, predictions: np.ndarray) -> np.ndarray:
        """
        Categorize predictions into risk strata based on thresholds.

        Example with thresholds [0.1, 0.3]:
            <0.1 → Category 0 (Low)
            0.1-0.3 → Category 1 (Intermediate)
            >0.3 → Category 2 (High)
        """
        categories = np.digitize(predictions, self.risk_thresholds)
        return categories

    def calculate_idi(self) -> float:
        """
        Calculate Integrated Discrimination Improvement.

        IDI = (IS_new - IS_base)
        where IS = Integrated Sensitivity = mean(pred|event) - mean(pred|nonevent)

        IDI measures improvement in average predicted risk separation
        between events and non-events.
        """
        # Integrated sensitivity for base model
        mean_pred_base_events = self.pred_base[self.y_true == 1].mean()
        mean_pred_base_nonevents = self.pred_base[self.y_true == 0].mean()
        is_base = mean_pred_base_events - mean_pred_base_nonevents

        # Integrated sensitivity for new model
        mean_pred_new_events = self.pred_new[self.y_true == 1].mean()
        mean_pred_new_nonevents = self.pred_new[self.y_true == 0].mean()
        is_new = mean_pred_new_events - mean_pred_new_nonevents

        # IDI
        idi = is_new - is_base

        return idi

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        method: str = 'categorical',
        random_seed: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for NRI and IDI.

        Args:
            n_bootstrap: Number of bootstrap iterations
            method: 'categorical' or 'continuous' for NRI
            random_seed: Random seed

        Returns:
            Dictionary with 95% CIs for NRI and IDI
        """
        np.random.seed(random_seed)

        nri_boots = []
        idi_boots = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(self.n, size=self.n, replace=True)

            y_boot = self.y_true[boot_idx]
            pred_base_boot = self.pred_base[boot_idx]
            pred_new_boot = self.pred_new[boot_idx]

            # Create temporary analyzer for bootstrap sample
            analyzer_boot = ReclassificationAnalyzer(
                y_boot,
                pred_base_boot,
                pred_new_boot,
                self.risk_thresholds
            )

            # Calculate metrics
            nri, _, _ = analyzer_boot.calculate_nri(method=method)
            idi = analyzer_boot.calculate_idi()

            nri_boots.append(nri)
            idi_boots.append(idi)

        # Calculate 95% CI (percentile method)
        nri_ci = (np.percentile(nri_boots, 2.5), np.percentile(nri_boots, 97.5))
        idi_ci = (np.percentile(idi_boots, 2.5), np.percentile(idi_boots, 97.5))

        return {
            'nri_ci': nri_ci,
            'idi_ci': idi_ci
        }

    def create_reclassification_table(self) -> pd.DataFrame:
        """
        Create reclassification table showing movement between risk categories.

        Rows: Base model risk categories
        Columns: New model risk categories
        """
        cat_base = self._categorize_risk(self.pred_base)
        cat_new = self._categorize_risk(self.pred_new)

        # Create category labels
        n_categories = len(self.risk_thresholds) + 1
        labels = []
        for i in range(n_categories):
            if i == 0:
                labels.append(f"Low (<{self.risk_thresholds[0]*100:.0f}%)")
            elif i == n_categories - 1:
                labels.append(f"High (≥{self.risk_thresholds[-1]*100:.0f}%)")
            else:
                labels.append(
                    f"Int ({self.risk_thresholds[i-1]*100:.0f}-{self.risk_thresholds[i]*100:.0f}%)"
                )

        # Create cross-tabulation
        reclass_table = pd.crosstab(
            cat_base,
            cat_new,
            rownames=['Base Model (AKIN)'],
            colnames=['New Model (AKIN + COSCO)'],
            margins=True
        )

        # Rename indices and columns
        reclass_table.index = labels + ['Total']
        reclass_table.columns = labels + ['Total']

        return reclass_table

    def perform_analysis(
        self,
        method: str = 'categorical',
        n_bootstrap: int = 1000
    ) -> ReclassificationResults:
        """
        Perform complete reclassification analysis.

        Args:
            method: 'categorical' or 'continuous' for NRI
            n_bootstrap: Number of bootstrap iterations for CIs

        Returns:
            ReclassificationResults object
        """
        # Calculate NRI
        nri, nri_events, nri_nonevents = self.calculate_nri(method=method)

        # Calculate IDI
        idi = self.calculate_idi()

        # Bootstrap confidence intervals
        ci_dict = self.bootstrap_ci(n_bootstrap=n_bootstrap, method=method)

        # Statistical significance (z-test)
        # NRI standard error (approximate)
        se_nri = np.sqrt((nri_events * (1 - nri_events) / self.n_events) +
                        (nri_nonevents * (1 - nri_nonevents) / self.n_nonevents))
        z_nri = nri / se_nri
        nri_pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_nri)))

        # IDI standard error (approximate)
        se_idi = np.sqrt(
            np.var(self.pred_new[self.y_true == 1]) / self.n_events +
            np.var(self.pred_new[self.y_true == 0]) / self.n_nonevents +
            np.var(self.pred_base[self.y_true == 1]) / self.n_events +
            np.var(self.pred_base[self.y_true == 0]) / self.n_nonevents
        )
        z_idi = idi / se_idi
        idi_pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_idi)))

        # Reclassification table
        reclass_table = self.create_reclassification_table()

        return ReclassificationResults(
            nri=nri,
            nri_events=nri_events,
            nri_nonevents=nri_nonevents,
            nri_pvalue=nri_pvalue,
            nri_ci_lower=ci_dict['nri_ci'][0],
            nri_ci_upper=ci_dict['nri_ci'][1],
            idi=idi,
            idi_pvalue=idi_pvalue,
            idi_ci_lower=ci_dict['idi_ci'][0],
            idi_ci_upper=ci_dict['idi_ci'][1],
            reclassification_table=reclass_table
        )

    def plot_reclassification(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize risk reclassification.

        Creates scatter plot showing base vs. new predicted risks,
        colored by outcome and risk category changes.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Scatter plot for events
        ax = axes[0]
        events_mask = (self.y_true == 1)
        ax.scatter(
            self.pred_base[events_mask],
            self.pred_new[events_mask],
            alpha=0.6,
            s=30,
            c='red',
            label='Events'
        )
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')

        # Add threshold lines
        for thresh in self.risk_thresholds:
            ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(thresh, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Base Model Risk (AKIN)', fontsize=11)
        ax.set_ylabel('New Model Risk (AKIN + COSCO)', fontsize=11)
        ax.set_title('Reclassification: Events', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Scatter plot for non-events
        ax = axes[1]
        nonevents_mask = (self.y_true == 0)
        ax.scatter(
            self.pred_base[nonevents_mask],
            self.pred_new[nonevents_mask],
            alpha=0.6,
            s=30,
            c='blue',
            label='Non-events'
        )
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')

        # Add threshold lines
        for thresh in self.risk_thresholds:
            ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(thresh, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Base Model Risk (AKIN)', fontsize=11)
        ax.set_ylabel('New Model Risk (AKIN + COSCO)', fontsize=11)
        ax.set_title('Reclassification: Non-events', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()


def main():
    """Example usage."""
    # Simulate example data
    np.random.seed(42)
    n = 382

    # True outcomes
    y_true = np.random.binomial(1, 0.27, n)

    # Base model predictions (AKIN) - weak signal
    pred_base = np.random.beta(2, 6, n)  # Mean ~0.25

    # New model predictions (AKIN + COSCO) - improved for events
    pred_new = pred_base.copy()
    # For events, increase predicted risk
    events_mask = (y_true == 1)
    pred_new[events_mask] += np.random.beta(2, 3, events_mask.sum()) * 0.2
    pred_new = np.clip(pred_new, 0, 1)

    print("="*70)
    print("RECLASSIFICATION ANALYSIS: COSCO ADDED VALUE")
    print("="*70)

    # Initialize analyzer
    analyzer = ReclassificationAnalyzer(
        y_true=y_true,
        pred_base=pred_base,
        pred_new=pred_new,
        risk_thresholds=[0.1, 0.3]  # Low, Intermediate, High
    )

    # Perform analysis
    results = analyzer.perform_analysis(method='categorical', n_bootstrap=1000)
    results.summary()

    # Print reclassification table
    print("\n" + "="*70)
    print("RECLASSIFICATION TABLE")
    print("="*70)
    print(results.reclassification_table)

    # Visualize
    analyzer.plot_reclassification(
        save_path='results/figures/reclassification_plot.png'
    )


if __name__ == '__main__':
    main()
