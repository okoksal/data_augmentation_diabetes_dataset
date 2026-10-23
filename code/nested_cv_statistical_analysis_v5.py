"""
nested_cv_statistical_analysis_v5.py

Companion analysis for imbalanced_v10_inner_cv.py. Reads <prefix>_nested_cv_detailed_results.csv
(one row per outer fold x augmentation x model) and writes every number the paper reports, so
that each statistic is traceable to code.

Usage:
    python nested_cv_statistical_analysis_v5.py pima_nested_cv_detailed_results.csv \
        [--outdir stats_v5] [--n-splits 3] [--n-boot 10000] [--seed 0]

IMPORTANT (read before quoting p-values)
The 45 outer folds come from repeated stratified 3-fold CV on ONE dataset, so they are
correlated. Three views of each paired contrast are therefore reported, from least to most
conservative:
  * fold-level Wilcoxon (n = 45)            -- ignores the dependence; optimistic.
  * repeat-level Wilcoxon / bootstrap (n = 15 repeat means) -- averages the 3 folds of a repeat,
    which removes the within-repeat dependence but not the dependence between repeats.
  * Nadeau-Bengio corrected resampled t-test -- variance inflated by (1/J + n_test/n_train);
    with 3-fold CV (ratio 0.5) it is very conservative.
Holm correction is applied per metric over the family of 15 within-classifier augmentation
contrasts (5 classifiers x 3 pairs). Everything except the pre-specified F1 family should be
described as exploratory.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ['f1', 'recall', 'specificity', 'false_negatives', 'auc_roc', 'pr_auc', 'brier_score']
AUG_PAIRS = [('smote', 'none'), ('wgan_gp', 'none'), ('wgan_gp', 'smote')]   # (A, B): reports A - B


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[idx]))
        adj[idx] = running
    return adj


def nadeau_bengio(diff, n_test_over_train):
    """Corrected resampled t-test on paired per-fold differences -> (se, p)."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    var = diff.var(ddof=1)
    se = np.sqrt(var * (1.0 / n + n_test_over_train))
    if se == 0:
        return 0.0, 1.0
    t = diff.mean() / se
    return se, float(2 * stats.t.sf(abs(t), n - 1))


def safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0):
        return 1.0
    return float(stats.wilcoxon(x).pvalue)


def paired_summary(diff, repeat_ids, n_test_over_train, rng, n_boot):
    """All views of one paired contrast. diff is indexed by fold (45 values)."""
    diff = pd.Series(diff).reset_index(drop=True)
    rep = diff.groupby(np.asarray(repeat_ids)).mean()
    fold_boot = rng.choice(diff.values, size=(n_boot, len(diff))).mean(axis=1)
    rep_boot = rng.choice(rep.values, size=(n_boot, len(rep))).mean(axis=1)
    se_nb, p_nb = nadeau_bengio(diff.values, n_test_over_train)
    return {
        'mean_diff': float(diff.mean()),
        'p_wilcoxon_fold': safe_wilcoxon(diff.values),
        'p_wilcoxon_repeat': safe_wilcoxon(rep.values),
        'p_nadeau_bengio': p_nb,
        'ci_fold_lo': float(np.percentile(fold_boot, 2.5)), 'ci_fold_hi': float(np.percentile(fold_boot, 97.5)),
        'ci_repeat_lo': float(np.percentile(rep_boot, 2.5)), 'ci_repeat_hi': float(np.percentile(rep_boot, 97.5)),
        'ci_nb_lo': float(diff.mean() - stats.t.ppf(0.975, len(diff) - 1) * se_nb),
        'ci_nb_hi': float(diff.mean() + stats.t.ppf(0.975, len(diff) - 1) * se_nb),
    }


def load(path, n_splits):
    d = pd.read_csv(path)
    need = {'fold', 'augmentation', 'model'} | set(METRICS)
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    d = d.sort_values(['fold', 'augmentation', 'model']).reset_index(drop=True)
    d['repeat'] = (d['fold'] - 1) // n_splits
    return d


def pivot(d, metric):
    return d.pivot_table(index='fold', columns=['model', 'augmentation'], values=metric)


def cell_summary(d):
    rows = []
    for (m, a), g in d.groupby(['model', 'augmentation']):
        r = {'model': m, 'augmentation': a, 'n_folds': len(g)}
        for k in METRICS:
            r[f'{k}_mean'] = g[k].mean()
            r[f'{k}_sd'] = g[k].std()
        r['f1_median'] = g['f1'].median()
        r['threshold_mean'] = g['threshold'].mean() if 'threshold' in g else np.nan
        r['collapsed_folds'] = int(g['collapsed'].sum()) if 'collapsed' in g else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values('f1_mean', ascending=False)


def augmentation_contrasts(d, n_test_over_train, rng, n_boot):
    """Within-classifier augmentation contrasts for every metric (Holm per metric over 15)."""
    models = sorted(d['model'].unique())
    out = []
    for metric in METRICS:
        P = pivot(d, metric)
        rep_ids = (P.index.values - 1) // int(round(1 / n_test_over_train + 1))   # n_splits = 1/ratio + 1
        rows = []
        for m in models:
            for a, b in AUG_PAIRS:
                s = paired_summary(P[(m, a)] - P[(m, b)], rep_ids, n_test_over_train, rng, n_boot)
                rows.append({'metric': metric, 'model': m, 'contrast': f'{a} - {b}', **s})
        df = pd.DataFrame(rows)
        df['p_holm_fold'] = holm(df['p_wilcoxon_fold'])
        df['p_holm_repeat'] = holm(df['p_wilcoxon_repeat'])
        out.append(df)
    return pd.concat(out, ignore_index=True)


def marginal_tests(d):
    """Average each metric over the 5 classifiers per fold, then Friedman over the 3 augmentations."""
    rows = []
    for metric in METRICS:
        P = pivot(d, metric)
        means = {a: P.xs(a, axis=1, level=1).mean(axis=1) for a in ['none', 'smote', 'wgan_gp']}
        chi = stats.friedmanchisquare(*[means[a].values for a in means])
        rows.append({'metric': metric, **{f'mean_{a}': float(v.mean()) for a, v in means.items()},
                     'friedman_chi2': float(chi.statistic), 'friedman_p': float(chi.pvalue)})
    return pd.DataFrame(rows)


def model_tests(d):
    """Friedman across classifiers within each augmentation (F1)."""
    rows = []
    P = pivot(d, 'f1')
    for a in ['none', 'smote', 'wgan_gp']:
        sub = P.xs(a, axis=1, level=1)
        if sub.shape[1] < 3:        # Friedman needs >= 3 related samples (classifiers)
            rows.append({'augmentation': a, 'friedman_chi2': np.nan, 'friedman_p': np.nan})
            continue
        chi = stats.friedmanchisquare(*[sub[m].values for m in sub.columns])
        rows.append({'augmentation': a, 'friedman_chi2': float(chi.statistic), 'friedman_p': float(chi.pvalue)})
    return pd.DataFrame(rows)


def top_combo_contrasts(d, n_test_over_train, rng, n_boot, metric='f1'):
    """Best-mean-F1 (model, augmentation) combination vs every other one, Holm over the family."""
    P = pivot(d, metric)
    best = P.mean().sort_values(ascending=False).index[0]
    rep_ids = (P.index.values - 1) // int(round(1 / n_test_over_train + 1))
    rows = []
    for col in P.columns:
        if col == best:
            continue
        s = paired_summary(P[best] - P[col], rep_ids, n_test_over_train, rng, n_boot)
        rows.append({'best': f'{best[0]}+{best[1]}', 'other': f'{col[0]}+{col[1]}', **s})
    df = pd.DataFrame(rows)
    df['p_holm_fold'] = holm(df['p_wilcoxon_fold'])
    df['p_holm_repeat'] = holm(df['p_wilcoxon_repeat'])
    return df.sort_values('p_wilcoxon_fold')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--outdir', default='stats_v5')
    ap.add_argument('--n-splits', type=int, default=3)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    d = load(a.csv, a.n_splits)
    ratio = 1.0 / (a.n_splits - 1)          # n_test / n_train for k-fold CV
    rng = np.random.RandomState(a.seed)
    n_folds = d['fold'].nunique()

    cells = cell_summary(d)
    contrasts = augmentation_contrasts(d, ratio, rng, a.n_boot)
    marg = marginal_tests(d)
    mods = model_tests(d)
    top = top_combo_contrasts(d, ratio, rng, a.n_boot)

    cells.to_csv(os.path.join(a.outdir, 'table4_cell_summary.csv'), index=False)
    contrasts.to_csv(os.path.join(a.outdir, 'augmentation_contrasts_all_metrics.csv'), index=False)
    marg.to_csv(os.path.join(a.outdir, 'marginal_augmentation_tests.csv'), index=False)
    mods.to_csv(os.path.join(a.outdir, 'friedman_models_within_augmentation.csv'), index=False)
    top.to_csv(os.path.join(a.outdir, 'best_f1_combo_vs_others.csv'), index=False)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    with open(os.path.join(a.outdir, 'report.txt'), 'w') as f:
        def w(t=''):
            print(t)
            f.write(t + '\n')
        w(f"# {os.path.basename(a.csv)}: {n_folds} outer folds, {d['repeat'].nunique()} repeats of {a.n_splits}-fold CV")
        w("# See the module docstring: fold-level p-values ignore dependence; repeat-level and Nadeau-Bengio do not.")
        w("\n## Cell summary (sorted by mean F1)")
        cols = ['model', 'augmentation', 'f1_mean', 'f1_median', 'f1_sd', 'recall_mean', 'specificity_mean',
                'false_negatives_mean', 'auc_roc_mean', 'pr_auc_mean', 'brier_score_mean', 'collapsed_folds']
        w(cells[[c for c in cols if c in cells.columns]].round(4).to_string(index=False))
        w("\n## Friedman across augmentations on the classifier-averaged metric")
        w(marg.round(4).to_string(index=False))
        w("\n## Friedman across classifiers within each augmentation (F1)")
        w(mods.round(4).to_string(index=False))
        for metric in ['f1', 'recall', 'specificity', 'false_negatives', 'pr_auc', 'auc_roc', 'brier_score']:
            w(f"\n## Within-classifier augmentation contrasts: {metric} (A - B)")
            sub = contrasts[contrasts['metric'] == metric].drop(columns='metric')
            w(sub.round(4).to_string(index=False))
        w("\n## Best mean-F1 combination vs every other combination (F1, A - B)")
        w(top.round(4).to_string(index=False))
    print(f"\nWrote report and CSVs to {a.outdir}/")


if __name__ == '__main__':
    main()
