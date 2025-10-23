"""
Data Simulation Module for COSCO Score Analysis

Generates realistic cohort data matching the characteristics of the
reported COSCO score validation study (N=382 AKI patients).

Key Features:
- AKIN staging (I, II, III) with realistic distribution
- COSCO components (hyperkalemia, sepsis, pulmonary edema, RRT)
- Time-to-event outcomes (ST_ACM and MACE)
- Realistic risk gradients matching reported mortality rates
- Covariate structure (age, comorbidities, baseline kidney function)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CohortParameters:
    """Parameters for realistic cohort simulation"""
    n_patients: int = 382
    random_seed: int = 42

    # AKIN distribution (approximately matching reported cohort)
    akin_dist: Tuple[float, float, float] = (0.35, 0.25, 0.40)  # I, II, III

    # Baseline characteristics
    age_mean: float = 68.0
    age_sd: float = 14.0
    diabetes_rate: float = 0.42
    ckd_rate: float = 0.35
    chf_rate: float = 0.28

    # COSCO component base rates (conditional on AKIN)
    hyperkalemia_base: Tuple[float, float, float] = (0.15, 0.25, 0.40)
    sepsis_base: Tuple[float, float, float] = (0.10, 0.15, 0.25)
    pulm_edema_base: Tuple[float, float, float] = (0.12, 0.20, 0.30)
    rrt_base: Tuple[float, float, float] = (0.05, 0.12, 0.28)

    # Outcome rates (target to match reported figures)
    # ST_ACM: COSCO 0→26.6%, COSCO 1→~40%, COSCO 2→~55%, COSCO 3→66.7%
    # MACE: Higher overall rates
    mortality_90d_base: float = 0.27  # Overall ~27%
    mace_90d_base: float = 0.45  # Overall ~45%

    # Follow-up censoring
    max_followup_days: int = 90
    censoring_rate: float = 0.05


class COSCODataSimulator:
    """
    Simulate realistic AKI cohort with AKIN stages, COSCO scores,
    and time-to-event outcomes.
    """

    def __init__(self, params: Optional[CohortParameters] = None):
        self.params = params or CohortParameters()
        np.random.seed(self.params.random_seed)

    def simulate_cohort(self) -> pd.DataFrame:
        """
        Generate complete synthetic cohort.

        Returns:
            DataFrame with columns:
                - patient_id
                - age, diabetes, ckd, chf (baseline characteristics)
                - akin_stage (1, 2, 3)
                - hyperkalemia, sepsis, pulm_edema, rrt (COSCO components)
                - cosco_score (0-4)
                - time_to_death, event_death (ST_ACM)
                - time_to_mace, event_mace (MACE)
                - complication_onset_time (for sensitivity analysis)
        """
        n = self.params.n_patients

        # Patient IDs
        patient_id = np.arange(1, n + 1)

        # Baseline characteristics
        age = np.random.normal(self.params.age_mean, self.params.age_sd, n)
        age = np.clip(age, 18, 95)  # Realistic age range

        diabetes = np.random.binomial(1, self.params.diabetes_rate, n)
        ckd = np.random.binomial(1, self.params.ckd_rate, n)
        chf = np.random.binomial(1, self.params.chf_rate, n)

        # AKIN staging
        akin_stage = np.random.choice(
            [1, 2, 3],
            size=n,
            p=self.params.akin_dist
        )

        # COSCO components (conditional on AKIN)
        hyperkalemia = self._simulate_component(
            akin_stage, self.params.hyperkalemia_base
        )
        sepsis = self._simulate_component(
            akin_stage, self.params.sepsis_base
        )
        pulm_edema = self._simulate_component(
            akin_stage, self.params.pulm_edema_base
        )
        rrt = self._simulate_component(
            akin_stage, self.params.rrt_base
        )

        # COSCO score (sum of components)
        cosco_score = hyperkalemia + sepsis + pulm_edema + rrt

        # Complication onset time (for time-aware analysis)
        # Early complications (day 1-3) vs late (day 4-14)
        complication_onset_time = np.where(
            cosco_score > 0,
            np.random.exponential(scale=4.0, size=n),  # Mean ~4 days
            np.nan
        )
        complication_onset_time = np.clip(complication_onset_time, 0, 14)

        # Simulate outcomes
        time_to_death, event_death = self._simulate_mortality(
            akin_stage, cosco_score, age, ckd, chf
        )

        time_to_mace, event_mace = self._simulate_mace(
            akin_stage, cosco_score, age, diabetes, chf
        )

        # Assemble dataframe
        df = pd.DataFrame({
            'patient_id': patient_id,
            'age': age,
            'diabetes': diabetes,
            'ckd': ckd,
            'chf': chf,
            'akin_stage': akin_stage,
            'hyperkalemia': hyperkalemia,
            'sepsis': sepsis,
            'pulm_edema': pulm_edema,
            'rrt': rrt,
            'cosco_score': cosco_score,
            'complication_onset_time': complication_onset_time,
            'time_to_death': time_to_death,
            'event_death': event_death,
            'time_to_mace': time_to_mace,
            'event_mace': event_mace
        })

        return df

    def _simulate_component(
        self,
        akin_stage: np.ndarray,
        base_rates: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Simulate binary complication with rates conditional on AKIN stage.

        Args:
            akin_stage: Array of AKIN stages (1, 2, 3)
            base_rates: Base probability for each AKIN stage (I, II, III)

        Returns:
            Binary array indicating presence of complication
        """
        component = np.zeros_like(akin_stage, dtype=int)

        for akin_val, rate in zip([1, 2, 3], base_rates):
            mask = (akin_stage == akin_val)
            n_akin = mask.sum()
            if n_akin > 0:
                component[mask] = np.random.binomial(1, rate, n_akin)

        return component

    def _simulate_mortality(
        self,
        akin_stage: np.ndarray,
        cosco_score: np.ndarray,
        age: np.ndarray,
        ckd: np.ndarray,
        chf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate 90-day mortality with realistic risk gradients.

        Target mortality rates by COSCO:
        - COSCO 0: ~26.6%
        - COSCO 1: ~40%
        - COSCO 2: ~55%
        - COSCO 3: ~66.7%

        AKIN shows flat profile (weak signal).

        Returns:
            time_to_death: Days to death (or censoring)
            event_death: 1 if death occurred, 0 if censored
        """
        n = len(akin_stage)

        # Compute log-hazard (Cox model simulation)
        # Weak AKIN effect (flat mortality profile)
        akin_ii = (akin_stage == 2).astype(float)
        akin_iii = (akin_stage == 3).astype(float)

        log_hazard = (
            -2.5 +  # Baseline log-hazard
            0.05 * akin_ii +  # Minimal AKIN II effect
            0.08 * akin_iii +  # Minimal AKIN III effect
            0.35 * cosco_score +  # Strong COSCO effect (0.35 per point)
            0.015 * (age - 68) +  # Age effect
            0.25 * ckd +  # CKD effect
            0.30 * chf +  # CHF effect
            np.random.normal(0, 0.3, n)  # Residual heterogeneity
        )

        # Weibull distribution for time-to-event
        shape = 1.2  # Slightly increasing hazard over time
        scale = np.exp(-log_hazard / shape)

        time_to_event = np.random.weibull(shape, n) * scale

        # Administrative censoring at 90 days
        time_to_censor = np.full(n, self.params.max_followup_days)

        # Random censoring (loss to follow-up)
        censor_uniform = np.random.uniform(0, self.params.max_followup_days, n)
        random_censor = np.where(
            np.random.uniform(0, 1, n) < self.params.censoring_rate,
            censor_uniform,
            self.params.max_followup_days
        )
        time_to_censor = np.minimum(time_to_censor, random_censor)

        # Observed time and event indicator
        time_to_death = np.minimum(time_to_event, time_to_censor)
        event_death = (time_to_event <= time_to_censor).astype(int)

        return time_to_death, event_death

    def _simulate_mace(
        self,
        akin_stage: np.ndarray,
        cosco_score: np.ndarray,
        age: np.ndarray,
        diabetes: np.ndarray,
        chf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate 90-day MACE (Major Adverse Cardiovascular Events).

        MACE has stronger signal for COSCO (AUC ~0.663).
        COSCO captures systemic instability → CV events.

        Returns:
            time_to_mace: Days to MACE (or censoring)
            event_mace: 1 if MACE occurred, 0 if censored
        """
        n = len(akin_stage)

        # Compute log-hazard
        # Moderate AKIN effect
        akin_ii = (akin_stage == 2).astype(float)
        akin_iii = (akin_stage == 3).astype(float)

        log_hazard = (
            -1.8 +  # Higher baseline (MACE more common)
            0.15 * akin_ii +  # Moderate AKIN II effect
            0.25 * akin_iii +  # Moderate AKIN III effect
            0.55 * cosco_score +  # STRONG COSCO effect (systemic instability)
            0.018 * (age - 68) +
            0.35 * diabetes +  # DM strong predictor of MACE
            0.40 * chf +  # CHF strong predictor
            np.random.normal(0, 0.35, n)
        )

        # Weibull distribution
        shape = 1.1
        scale = np.exp(-log_hazard / shape)

        time_to_event = np.random.weibull(shape, n) * scale

        # Censoring (same logic as mortality)
        time_to_censor = np.full(n, self.params.max_followup_days)
        censor_uniform = np.random.uniform(0, self.params.max_followup_days, n)
        random_censor = np.where(
            np.random.uniform(0, 1, n) < self.params.censoring_rate,
            censor_uniform,
            self.params.max_followup_days
        )
        time_to_censor = np.minimum(time_to_censor, random_censor)

        time_to_mace = np.minimum(time_to_event, time_to_censor)
        event_mace = (time_to_event <= time_to_censor).astype(int)

        return time_to_mace, event_mace

    def generate_and_save(self, output_path: str = 'data/simulated_cohort.csv'):
        """
        Generate cohort and save to CSV.

        Args:
            output_path: Path to save CSV file
        """
        df = self.simulate_cohort()
        df.to_csv(output_path, index=False)
        print(f"Generated cohort: N={len(df)} patients")
        print(f"Saved to: {output_path}")

        # Print summary statistics
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print cohort summary statistics."""
        print("\n" + "="*60)
        print("COHORT SUMMARY STATISTICS")
        print("="*60)

        print(f"\nBaseline Characteristics:")
        print(f"  Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
        print(f"  Diabetes: {df['diabetes'].mean()*100:.1f}%")
        print(f"  CKD: {df['ckd'].mean()*100:.1f}%")
        print(f"  CHF: {df['chf'].mean()*100:.1f}%")

        print(f"\nAKIN Distribution:")
        for stage in [1, 2, 3]:
            n = (df['akin_stage'] == stage).sum()
            pct = n / len(df) * 100
            print(f"  Stage {stage}: {n} ({pct:.1f}%)")

        print(f"\nCOSCO Components:")
        print(f"  Hyperkalemia: {df['hyperkalemia'].sum()} ({df['hyperkalemia'].mean()*100:.1f}%)")
        print(f"  Sepsis: {df['sepsis'].sum()} ({df['sepsis'].mean()*100:.1f}%)")
        print(f"  Pulmonary Edema: {df['pulm_edema'].sum()} ({df['pulm_edema'].mean()*100:.1f}%)")
        print(f"  RRT: {df['rrt'].sum()} ({df['rrt'].mean()*100:.1f}%)")

        print(f"\nCOSCO Score Distribution:")
        for score in range(5):
            n = (df['cosco_score'] == score).sum()
            if n > 0:
                mort_rate = df[df['cosco_score'] == score]['event_death'].mean() * 100
                mace_rate = df[df['cosco_score'] == score]['event_mace'].mean() * 100
                print(f"  Score {score}: {n} pts | Mortality: {mort_rate:.1f}% | MACE: {mace_rate:.1f}%")

        print(f"\nOutcomes (90-day):")
        print(f"  Deaths: {df['event_death'].sum()} ({df['event_death'].mean()*100:.1f}%)")
        print(f"  MACE: {df['event_mace'].sum()} ({df['event_mace'].mean()*100:.1f}%)")
        print(f"  Median follow-up: {df['time_to_death'].median():.1f} days")

        print(f"\n" + "="*60)


def main():
    """Example usage."""
    simulator = COSCODataSimulator()
    df = simulator.generate_and_save('data/simulated_cohort.csv')

    # Optional: generate multiple datasets for sensitivity analysis
    print("\n\nGenerating additional validation cohorts...")
    for i in range(3):
        params = CohortParameters(
            n_patients=382,
            random_seed=42 + i + 1
        )
        simulator = COSCODataSimulator(params)
        simulator.generate_and_save(f'data/validation_cohort_{i+1}.csv')


if __name__ == '__main__':
    main()
