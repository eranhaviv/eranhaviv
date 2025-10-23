# COSCO Score Analysis - Quick Start Guide

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/eranhaviv/eranhaviv.git
cd eranhaviv
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cosco python=3.9
conda activate cosco
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Analysis

### Option 1: Complete Analysis with Simulated Data (Fastest Start)

```bash
cd src
python comprehensive_analysis.py
```

This will:
- Generate realistic simulated cohort (N=382 patients)
- Run all 5 mandated statistical analyses
- Generate publication-ready figures
- Create summary report

**Output:**
- `results/analysis_summary.txt` - Complete statistical summary
- `results/figures/` - All plots (KM curves, DCA, calibration, etc.)
- `results/tables/` - Statistical tables (reclassification, phenotypes)

**Time:** ~5-10 minutes (depending on bootstrap iterations)

### Option 2: Analysis with Your Own Data

Prepare CSV file with these columns:
```
patient_id, age, diabetes, ckd, chf,
akin_stage (1/2/3),
hyperkalemia (0/1), sepsis (0/1), pulm_edema (0/1), rrt (0/1),
time_to_death, event_death (0/1),
time_to_mace, event_mace (0/1)
```

Then run:
```python
from src.comprehensive_analysis import COSCOAnalysis

analyzer = COSCOAnalysis(
    data_path='path/to/your_data.csv',
    n_bootstrap=1000,
    output_dir='results'
)

results = analyzer.run_complete_analysis()
```

## Individual Module Usage

### 1. Generate Simulated Data Only
```python
from src.data_simulation import COSCODataSimulator

simulator = COSCODataSimulator()
df = simulator.generate_and_save('data/my_cohort.csv')
```

### 2. Bootstrap Validation Only
```python
from src.bootstrap_validation import BootstrapValidator
import pandas as pd

df = pd.read_csv('data/simulated_cohort.csv')
validator = BootstrapValidator(n_iterations=1000)

# Prepare data
X = df[['cosco_score', 'age', 'ckd', 'chf']]
y = df['event_death'].values

# Validate
result = validator.validate_logistic(X, y)
print(result)
# Output: Corrected C-statistic with 95% CI
```

### 3. Cox Proportional Hazards
```python
from src.survival_models import SurvivalAnalyzer

df = pd.read_csv('data/simulated_cohort.csv')
analyzer = SurvivalAnalyzer(df)

# Fit all models (AKIN, COSCO, Combined)
results = analyzer.fit_all_models(outcome='mortality')

# Generate Kaplan-Meier curves
analyzer.plot_kaplan_meier(
    stratify_by='cosco_cat',
    outcome='mortality',
    save_path='km_curve.png'
)
```

### 4. Reclassification Metrics (NRI/IDI)
```python
from src.reclassification_metrics import ReclassificationAnalyzer

# Assume you have predictions from base and new models
analyzer = ReclassificationAnalyzer(
    y_true=y,
    pred_base=predictions_akin,
    pred_new=predictions_combined,
    risk_thresholds=[0.1, 0.3]  # Low/Intermediate/High
)

results = analyzer.perform_analysis(method='categorical')
results.summary()
# Output: NRI, IDI with p-values and CIs
```

### 5. Calibration Assessment
```python
from src.calibration_analysis import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(
    y_true=y,
    y_pred=predicted_probabilities,
    n_bins=10
)

cal_results = analyzer.perform_analysis()
cal_results.summary()
# Output: Hosmer-Lemeshow test, calibration slope, Brier score

analyzer.plot_calibration(save_path='calibration.png')
```

### 6. Decision Curve Analysis
```python
from src.decision_curve_analysis import compare_cosco_models_dca

df = pd.read_csv('data/simulated_cohort.csv')

dca_results, analyzer = compare_cosco_models_dca(
    df,
    outcome='mortality'
)

# Plot decision curves
dca_results.plot(save_path='dca.png')

# Compare at specific threshold
comparison = analyzer.compare_models_at_threshold(threshold=0.30)
print(comparison)
```

## Understanding the Output

### Key Metrics Explained

**1. Bias-Corrected C-Statistic**
- Apparent AUC: Performance on training data (optimistic)
- Optimism: Amount of overfitting detected via bootstrap
- **Corrected AUC**: Reliable estimate after removing optimism
- Target: MACE should show C-statistic ~0.66 for COSCO

**2. Hazard Ratios (HR) from Cox Models**
- HR > 1: Increased risk of event
- HR < 1: Decreased risk
- Example: COSCO score HR = 1.5 means 50% increased hazard per point

**3. Net Reclassification Improvement (NRI)**
- Measures % of patients correctly reclassified
- NRI > 0: COSCO improves risk stratification
- p < 0.05: Statistically significant improvement

**4. Calibration**
- Good calibration: Hosmer-Lemeshow p > 0.05
- Calibration slope ~1.0 ideal
- Brier score: Lower is better (max = 0.25)

**5. Decision Curve Analysis**
- Net Benefit > 0: Using model helps patients
- Compare to "Treat All" and "Treat None"
- Optimal threshold: Highest net benefit

## Typical Analysis Workflow

```bash
# Step 1: Generate or prepare data
python -c "from src.data_simulation import COSCODataSimulator; \
           COSCODataSimulator().generate_and_save('data/cohort.csv')"

# Step 2: Run comprehensive analysis
cd src
python comprehensive_analysis.py

# Step 3: Review results
cat ../results/analysis_summary.txt
ls ../results/figures/
ls ../results/tables/

# Step 4: Update manuscript with corrected statistics
# Use bias-corrected AUCs from analysis_summary.txt
```

## Customization

### Adjust Bootstrap Iterations
```python
analyzer = COSCOAnalysis(
    data_path='data.csv',
    n_bootstrap=500,  # Reduce for speed (min 500)
    output_dir='results'
)
```

### Change Risk Thresholds for NRI
```python
analyzer = ReclassificationAnalyzer(
    y_true=y,
    pred_base=pred_base,
    pred_new=pred_new,
    risk_thresholds=[0.15, 0.35]  # Custom thresholds
)
```

### Modify Cohort Simulation Parameters
```python
from src.data_simulation import CohortParameters

params = CohortParameters(
    n_patients=500,  # Larger cohort
    random_seed=123,
    mortality_90d_base=0.30  # Higher mortality
)

simulator = COSCODataSimulator(params)
df = simulator.simulate_cohort()
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Ensure you're in the correct directory and installed dependencies
```bash
cd eranhaviv
pip install -r requirements.txt
```

### Issue: "Convergence warning in Cox model"
**Solution:** This is often harmless, but you can increase penalization:
```python
result = analyzer.fit_cox_model(covariates, penalizer=0.01)
```

### Issue: Bootstrap is too slow
**Solution:** Reduce iterations or use fewer parallel jobs:
```python
validator = BootstrapValidator(n_iterations=500, n_jobs=4)
```

### Issue: Calibration plot looks poor
**Solution:** This may indicate true poor calibration. Check:
- Sample size adequate?
- Extreme outliers in predictions?
- Model overfitting?

## Citation

If you use this framework, please cite:

```bibtex
@software{cosco_analysis_2025,
  author = {[Your Name]},
  title = {COSCO Score Analysis Framework: Rigorous Statistical Validation},
  year = {2025},
  url = {https://github.com/eranhaviv/eranhaviv}
}
```

## Support

- **Documentation**: See [COSCO_ANALYSIS_README.md](COSCO_ANALYSIS_README.md)
- **Issues**: https://github.com/eranhaviv/eranhaviv/issues
- **Examples**: Check `src/` for docstrings in each module

---

**Last Updated:** 2025-10-23
