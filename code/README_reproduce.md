# Reproducing the Pima results

Environment: see `requirements_pinned_v10b.txt` (Python 3.11, CPU). Use the same XGBoost version.

1. Tests
   pytest test_imbalanced_v10b_inner_cv.py test_nested_cv_statistical_analysis_v5.py test_calibration_analysis_v1.py -v
2. Main benchmark (about 4 hours on CPU; writes into runs/<timestamp>/)
   python imbalanced_v10b_inner_cv.py --datasets pima --deterministic --save-predictions
3. Statistical analysis (every p-value, CI and table in the paper)
   python nested_cv_statistical_analysis_v5.py pima_nested_cv_detailed_results.csv --outdir stats_v5
4. Calibration analysis (pooled reliability diagrams, slope/intercept)
   python calibration_analysis_v1.py pima_outer_test_predictions.csv.gz --outdir calib_v1

Data: pima-indians-diabetes.data.csv from github.com/jbrownlee/Datasets (cached as diabetes.csv),
MD5 56a8d8ae619fcc223941e54f361b8406.
Use pima_nested_cv_detailed_results.csv (not the rounded predictions file) for AUC and PR-AUC.

## Sensitivity analyses reported in the paper

5. Without treating zeros as missing (same code, same seed):
   python imbalanced_v10b_inner_cv.py --datasets pima --deterministic --no-impute
6. Earlier, more permissive protocol (hyperparameters searched on augmented data and scored by F1 at 0.5,
   threshold from a single 25% hold-out applied to a separately trained model, WGAN-GP configuration
   selected on that hold-out). Script: imbalanced_v9_perfold_wgan.py, run with:
   python imbalanced_v9_perfold_wgan.py --datasets pima --deterministic
   (its results were produced in an earlier environment: LightGBM 4.7.0, XGBoost 3.2.0, GPU; the SVM
   results compared in the paper do not depend on LightGBM.)
Analyse each run's *_nested_cv_detailed_results.csv with nested_cv_statistical_analysis_v5.py.
