# e-lightgbm-lol-skilltier
# E-LightGBM: Focal-Loss-Engineered, SHAP-Pruned LightGBM for League of Legends Skill-Tier Prediction

Research code for an undergraduate/postgraduate ML research proposal (Department of Computer Science, KNUST) predicting individual League of Legends player skill tier from in-match performance statistics, using a focal-loss-engineered and SHAP-pruned variant of LightGBM (**E-LightGBM**).

## Project Status

🚧 Research in progress — proposal stage. Baseline and engineered model implementation to follow the phase workflow below.

## Research Summary

- **Task:** Multiclass classification of player skill tier (9 classes, Iron–Challenger) from in-match performance statistics.
- **Engineering contribution:**
  - **[F] Fabricate** — custom multiclass focal-loss training objective (replacing default multiclass log-loss) to up-weight rare elite tiers.
  - **[R] Remove** — SHAP-guided pruning of the bottom 30% of features by mean absolute SHAP value.
- **Baseline:** Locked, unmodified multiclass LightGBM classifier, trained on identical data splits/seed, evaluated before any engineering.
- **Comparison:** Accuracy, Macro-F1, AUC-ROC, AUC-PR (per-class and macro), training time; Wilcoxon Signed-Rank significance test across 5-fold cross-validation.
- **Data:** Hybrid — public Kaggle/Riot-API League of Legends match dataset (training/cross-validation) + a freshly pulled field sample via the Riot Games Developer API (held-out generalization/temporal-transfer test set).

## Repository Structure

```
.
├── data/                   # Raw and processed data (not committed; see .gitignore)
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_baseline.ipynb       # Phase 1: locked, unmodified LightGBM baseline
│   ├── 02_engineered.ipynb     # Phase 2: E-LightGBM (focal loss + SHAP pruning)
│   └── 03_comparison.ipynb     # Phase 3: formal baseline-vs-engineered comparison
├── src/
│   ├── preprocessing.py
│   ├── focal_loss.py            # Custom multiclass focal-loss objective/gradient/Hessian
│   ├── shap_pruning.py
│   └── evaluation.py
├── results/
│   ├── baseline_metrics.json    # Locked baseline metrics (never modified after Phase 1)
│   └── comparison_table.csv     # Final baseline-vs-engineered results table
├── data_dictionary.csv          # Full variable manifest (see Section 4C of the proposal)
├── requirements.txt
└── README.md
```

## Reproducibility

- **Random seed:** `SEED = 42`, fixed across data splitting, LightGBM training, and SHAP sampling.
- **Environment:** Kaggle Notebooks (Free tier) for development; Google Colab Pro as fallback for longer SHAP runs. CPU-only training is sufficient at this data scale.
- **Software versions:** see `requirements.txt` (pinned).
- **Data splits:** Stratified 70/15/15 train/val/test split by skill tier; 5-fold stratified cross-validation on the training set.

## Phase Workflow

1. **Phase 0** — Set up a clean, reproducible environment (virtual env, pinned package versions).
2. **Phase 1** — Build and lock the unmodified LightGBM baseline; save its metrics; never touch it again.
3. **Phase 2** — Implement the focal-loss objective and SHAP-guided pruning; build E-LightGBM in a separate notebook on identical data splits/seed.
4. **Phase 3** — Formal comparison: Accuracy, Macro-F1, AUC-ROC, AUC-PR, training time, plus a Wilcoxon Signed-Rank significance test and effect size, reported in one results table.

## Citation

If you use this code, please cite the associated proposal/paper (citation details to be added on publication).

## License

MIT License (or institutional default — update before publishing).

## Contact

[Nbang-ba Suabir Abdul-Kahar Adam] — Department of Computer Science, KNUST. 
