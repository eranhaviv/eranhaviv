"""
Decision Curve Analysis (DCA) Module

Implements decision curve analysis to quantify clinical net benefit
of prognostic models across a range of risk thresholds.

DCA is the gold standard for evaluating clinical utility beyond
discrimination metrics (AUC, NRI). It answers: "Does using this model
to guide decisions improve patient outcomes?"

Key Features:
- Net benefit calculation across risk thresholds
- Comparison of multiple models and strategies
- Visualization of decision curves
- Optimal threshold identification
- Cost-benefit analysis

References:
Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
evaluating prediction models. Med Decis Making. 2006.

Fitzgerald M, Saville BR, Lewis RJ. Decision curve analysis.
JAMA. 2015.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DCAResults:
    """Results from decision curve analysis."""
    thresholds: np.ndarray
    net_benefit: Dict[str, np.ndarray]
    optimal_threshold: float
    optimal_net_benefit: float

    def plot(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """Plot decision curves."""
        fig, ax = plt.subplots(figsize=figsize)

        colors = sns.color_palette("husl", len(self.net_benefit))

        for i, (model_name, nb) in enumerate(self.net_benefit.items()):
            if model_name in ['Treat All', 'Treat None']:
                linestyle = '--'
                linewidth = 2
                alpha = 0.6
            else:
                linestyle = '-'
                linewidth = 2.5
                alpha = 1.0

            ax.plot(
                self.thresholds,
                nb,
                label=model_name,
                linestyle=linestyle,
                linewidth=linewidth,
                color=colors[i],
                alpha=alpha
            )

        # Highlight optimal threshold
        if self.optimal_threshold is not None:
            ax.axvline(
                self.optimal_threshold,
                color='red',
                linestyle=':',
                linewidth=2,
                alpha=0.7,
                label=f'Optimal threshold: {self.optimal_threshold:.2f}'
            )

        ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
        ax.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, max(self.thresholds)])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()


class DecisionCurveAnalyzer:
    """
    Decision curve analysis for prognostic models.

    Core Concept:
    Net Benefit = (True Positives / N) - (False Positives / N) × (odds at threshold)

    Where odds = threshold / (1 - threshold)

    Interpretation:
    - Net benefit > 0: Using model provides benefit
    - Compare model to "Treat All" and "Treat None" strategies
    - Higher net benefit = better clinical utility
    """

    def __init__(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        thresholds: Optional[np.ndarray] = None
    ):
        """
        Initialize decision curve analyzer.

        Args:
            y_true: True binary outcomes (0/1)
            predictions: Dictionary of model predictions
                        {model_name: predicted_probabilities}
            thresholds: Array of threshold probabilities to evaluate
                       If None, uses np.linspace(0.01, 0.99, 100)
        """
        self.y_true = np.array(y_true)
        self.predictions = predictions
        self.n = len(y_true)
        self.prevalence = y_true.mean()

        if thresholds is None:
            self.thresholds = np.linspace(0.01, 0.99, 100)
        else:
            self.thresholds = thresholds

    def calculate_net_benefit(
        self,
        y_pred: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate net benefit at a specific threshold.

        Algorithm:
        1. Classify as "high risk" if predicted_prob ≥ threshold
        2. True Positives (TP): High risk AND event occurred
        3. False Positives (FP): High risk AND no event
        4. Net Benefit = TP/N - FP/N × (threshold / (1 - threshold))

        Args:
            y_pred: Predicted probabilities
            threshold: Risk threshold for classification

        Returns:
            Net benefit (can be negative)
        """
        # Classify as high risk
        high_risk = (y_pred >= threshold).astype(int)

        # True positives and false positives
        tp = np.sum((high_risk == 1) & (self.y_true == 1))
        fp = np.sum((high_risk == 1) & (self.y_true == 0))

        # Calculate net benefit
        tp_rate = tp / self.n
        fp_rate = fp / self.n

        # Odds at threshold
        odds = threshold / (1 - threshold)

        net_benefit = tp_rate - fp_rate * odds

        return net_benefit

    def calculate_treat_all(self, threshold: float) -> float:
        """
        Calculate net benefit for "Treat All" strategy.

        Assume all patients are treated (high risk) regardless of model.

        Args:
            threshold: Risk threshold

        Returns:
            Net benefit of treating all
        """
        # All patients classified as high risk
        # TP = all events, FP = all non-events

        tp_rate = self.prevalence
        fp_rate = 1 - self.prevalence

        odds = threshold / (1 - threshold)

        net_benefit = tp_rate - fp_rate * odds

        return net_benefit

    def calculate_treat_none(self) -> float:
        """
        Calculate net benefit for "Treat None" strategy.

        Assume no patients are treated (all low risk).

        Returns:
            Net benefit of treating none (always 0)
        """
        return 0.0

    def perform_analysis(self) -> DCAResults:
        """
        Perform decision curve analysis for all models.

        Returns:
            DCAResults object with net benefits for each model
        """
        net_benefits = {}

        # Calculate net benefit for each model
        for model_name, y_pred in self.predictions.items():
            nb_curve = []

            for threshold in self.thresholds:
                nb = self.calculate_net_benefit(y_pred, threshold)
                nb_curve.append(nb)

            net_benefits[model_name] = np.array(nb_curve)

        # Calculate treat all strategy
        treat_all_curve = []
        for threshold in self.thresholds:
            nb = self.calculate_treat_all(threshold)
            treat_all_curve.append(nb)

        net_benefits['Treat All'] = np.array(treat_all_curve)

        # Calculate treat none strategy (always 0)
        net_benefits['Treat None'] = np.zeros_like(self.thresholds)

        # Find optimal threshold (for primary model)
        # Use first non-reference model
        primary_model = list(self.predictions.keys())[0]
        primary_nb = net_benefits[primary_model]

        # Optimal = highest net benefit
        optimal_idx = np.argmax(primary_nb)
        optimal_threshold = self.thresholds[optimal_idx]
        optimal_net_benefit = primary_nb[optimal_idx]

        return DCAResults(
            thresholds=self.thresholds,
            net_benefit=net_benefits,
            optimal_threshold=optimal_threshold,
            optimal_net_benefit=optimal_net_benefit
        )

    def compare_models_at_threshold(
        self,
        threshold: float
    ) -> pd.DataFrame:
        """
        Compare net benefits of all models at a specific threshold.

        Useful for clinical decision-making at a pre-specified
        risk threshold (e.g., 30% mortality risk).

        Args:
            threshold: Risk threshold

        Returns:
            DataFrame with net benefits for each model
        """
        comparison = []

        for model_name, y_pred in self.predictions.items():
            nb = self.calculate_net_benefit(y_pred, threshold)

            comparison.append({
                'Model': model_name,
                'Net Benefit': nb,
                'Threshold': threshold
            })

        # Add treat all and treat none
        comparison.append({
            'Model': 'Treat All',
            'Net Benefit': self.calculate_treat_all(threshold),
            'Threshold': threshold
        })

        comparison.append({
            'Model': 'Treat None',
            'Net Benefit': 0.0,
            'Threshold': threshold
        })

        df = pd.DataFrame(comparison)
        df = df.sort_values('Net Benefit', ascending=False)

        return df

    def calculate_intervention_metrics(
        self,
        y_pred: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """
        Calculate intervention-related metrics at a threshold.

        Useful for cost-benefit and resource planning.

        Args:
            y_pred: Predicted probabilities
            threshold: Risk threshold

        Returns:
            Dictionary with:
                - n_treated: Number classified as high risk
                - n_true_positives: Events correctly identified
                - n_false_positives: Non-events incorrectly flagged
                - ppv: Positive Predictive Value
                - npv: Negative Predictive Value
                - number_needed_to_treat: NNT
        """
        # Classify
        high_risk = (y_pred >= threshold).astype(int)

        # Counts
        tp = np.sum((high_risk == 1) & (self.y_true == 1))
        fp = np.sum((high_risk == 1) & (self.y_true == 0))
        tn = np.sum((high_risk == 0) & (self.y_true == 0))
        fn = np.sum((high_risk == 0) & (self.y_true == 1))

        n_treated = tp + fp

        # PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Number needed to treat (inverse of absolute risk reduction)
        # Simplified: NNT = 1 / PPV
        nnt = 1 / ppv if ppv > 0 else np.inf

        return {
            'n_treated': n_treated,
            'n_true_positives': tp,
            'n_false_positives': fp,
            'ppv': ppv,
            'npv': npv,
            'number_needed_to_treat': nnt
        }


def compare_cosco_models_dca(
    df: pd.DataFrame,
    outcome: str = 'mortality'
) -> DCAResults:
    """
    Decision curve analysis comparing AKIN, COSCO, and combined models.

    Args:
        df: Cohort dataframe
        outcome: 'mortality' or 'mace'

    Returns:
        DCAResults object
    """
    from sklearn.linear_model import LogisticRegression

    # Prepare data
    if outcome == 'mortality':
        y = df['event_death'].values
    else:
        y = df['event_mace'].values

    # Create dummy variables
    df['akin_ii'] = (df['akin_stage'] == 2).astype(int)
    df['akin_iii'] = (df['akin_stage'] == 3).astype(int)

    # Baseline covariates
    baseline = ['age', 'ckd', 'chf']
    if outcome == 'mace':
        baseline.append('diabetes')

    # Model 1: AKIN alone
    X_akin = df[['akin_ii', 'akin_iii'] + baseline]
    model_akin = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model_akin.fit(X_akin, y)
    pred_akin = model_akin.predict_proba(X_akin)[:, 1]

    # Model 2: COSCO alone
    X_cosco = df[['cosco_score'] + baseline]
    model_cosco = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model_cosco.fit(X_cosco, y)
    pred_cosco = model_cosco.predict_proba(X_cosco)[:, 1]

    # Model 3: Combined
    X_combined = df[['akin_ii', 'akin_iii', 'cosco_score'] + baseline]
    model_combined = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model_combined.fit(X_combined, y)
    pred_combined = model_combined.predict_proba(X_combined)[:, 1]

    # Decision curve analysis
    predictions = {
        'AKIN': pred_akin,
        'COSCO': pred_cosco,
        'AKIN + COSCO': pred_combined
    }

    analyzer = DecisionCurveAnalyzer(
        y_true=y,
        predictions=predictions,
        thresholds=np.linspace(0.01, 0.75, 100)
    )

    results = analyzer.perform_analysis()

    return results, analyzer


def main():
    """Example usage."""
    # Load data
    df = pd.read_csv('data/simulated_cohort.csv')

    print("="*70)
    print("DECISION CURVE ANALYSIS: 90-DAY MORTALITY")
    print("="*70)

    # Perform DCA for mortality
    results_mort, analyzer_mort = compare_cosco_models_dca(df, outcome='mortality')

    print(f"\nOptimal threshold: {results_mort.optimal_threshold:.2f}")
    print(f"Net benefit at optimal: {results_mort.optimal_net_benefit:.4f}")

    # Compare at clinically relevant threshold (30% mortality risk)
    print("\n" + "="*70)
    print("NET BENEFIT AT 30% MORTALITY THRESHOLD")
    print("="*70)
    comparison = analyzer_mort.compare_models_at_threshold(threshold=0.30)
    print(comparison.to_string(index=False))

    # Intervention metrics
    print("\n" + "="*70)
    print("INTERVENTION METRICS (Combined Model at 30% threshold)")
    print("="*70)
    from sklearn.linear_model import LogisticRegression

    df['akin_ii'] = (df['akin_stage'] == 2).astype(int)
    df['akin_iii'] = (df['akin_stage'] == 3).astype(int)
    X = df[['akin_ii', 'akin_iii', 'cosco_score', 'age', 'ckd', 'chf']]
    y = df['event_death'].values
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model.fit(X, y)
    pred = model.predict_proba(X)[:, 1]

    metrics = analyzer_mort.calculate_intervention_metrics(pred, threshold=0.30)

    print(f"\nPatients classified as high risk: {metrics['n_treated']}")
    print(f"True positives (events identified): {metrics['n_true_positives']}")
    print(f"False positives (unnecessary intervention): {metrics['n_false_positives']}")
    print(f"Positive Predictive Value: {metrics['ppv']:.3f}")
    print(f"Negative Predictive Value: {metrics['npv']:.3f}")
    print(f"Number Needed to Treat: {metrics['number_needed_to_treat']:.1f}")

    # Plot decision curves
    print("\n" + "="*70)
    print("GENERATING DECISION CURVE PLOTS")
    print("="*70)
    results_mort.plot(save_path='results/figures/dca_mortality.png')

    # MACE analysis
    print("\n" + "="*70)
    print("DECISION CURVE ANALYSIS: MACE")
    print("="*70)
    results_mace, analyzer_mace = compare_cosco_models_dca(df, outcome='mace')
    results_mace.plot(save_path='results/figures/dca_mace.png')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
