"""
calibration_analysis_v1.py

Pooled calibration analysis of the outer-test predictions written by
imbalanced_v10b_inner_cv.py --save-predictions  (<prefix>_outer_test_predictions.csv.gz).

Every repeat of the repeated k-fold CV scores each patient exactly once, so within each repeat the
k outer-test folds are pooled into one out-of-sample prediction per patient (768 for Pima). For every
(classifier, augmentation, repeat) we compute
  * mean predicted probability vs observed prevalence  (calibration-in-the-large gap),
  * calibration slope  (logistic regression of the outcome on logit(p); 1 = ideal; <1 = overconfident),
  * calibration intercept at slope 1 (offset model; 0 = ideal; <0 = probabilities too high),
  * Brier score.
These are summarised over the repeats (mean, SD). The reliability diagrams pool all repeats.
Contrasts versus 'none' use the exact Wilcoxon test on the repeat-level differences (n = number of repeats);
treat them as exploratory (the repeats are not independent: they reuse the same patients).

Usage: python calibration_analysis_v1.py pima_outer_test_predictions.csv.gz [--outdir calib_v1] [--n-splits 3] [--bins 10]
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
AUG_LABEL = {'none': 'No augmentation', 'smote': 'SMOTE', 'wgan_gp': 'WGAN-GP'}
AUG_COLOR = {'none': '#4C72B0', 'smote': '#DD8452', 'wgan_gp': '#55A868'}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calibration_slope(y, p):
    """Slope of the logistic recalibration y ~ a + b * logit(p) (unpenalised)."""
    lr = LogisticRegression(C=1e10, max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def calibration_intercept(y, p, iters=50):
    """Intercept a of the offset model y ~ a + 1 * logit(p) (calibration-in-the-large at slope 1)."""
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    a = 0.0
    for _ in range(iters):
        mu = sigmoid(lp + a)
        step = (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def per_repeat_metrics(pred, n_splits):
    pred = pred.copy()
    pred['repeat'] = (pred['fold'] - 1) // n_splits
    rows = []
    for (m, a, r), g in pred.groupby(['model', 'augmentation', 'repeat']):
        y, p = g['y_true'].values, g['prob'].values
        if len(np.unique(y)) < 2:
            continue
        rows.append({'model': m, 'augmentation': a, 'repeat': r, 'n': len(g),
                     'mean_pred': p.mean(), 'prevalence': y.mean(),
                     'citl_gap': p.mean() - y.mean(),
                     'slope': calibration_slope(y, p), 'intercept': calibration_intercept(y, p),
                     'brier': float(np.mean((p - y) ** 2))})
    return pd.DataFrame(rows)


def summarise(rep):
    agg = rep.groupby(['model', 'augmentation']).agg(
        n_repeats=('repeat', 'nunique'), mean_pred=('mean_pred', 'mean'), prevalence=('prevalence', 'mean'),
        citl_gap=('citl_gap', 'mean'), slope=('slope', 'mean'), slope_sd=('slope', 'std'),
        intercept=('intercept', 'mean'), intercept_sd=('intercept', 'std'),
        brier=('brier', 'mean'), brier_sd=('brier', 'std')).reset_index()
    return agg


def contrasts_vs_none(rep):
    rows = []
    for m in sorted(rep['model'].unique()):
        base = rep[(rep.model == m) & (rep.augmentation == 'none')].set_index('repeat')
        for a in ['smote', 'wgan_gp']:
            cur = rep[(rep.model == m) & (rep.augmentation == a)].set_index('repeat')
            common = base.index.intersection(cur.index)
            for metric in ['citl_gap', 'slope', 'intercept', 'brier']:
                d = (cur.loc[common, metric] - base.loc[common, metric]).values
                p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue)
                rows.append({'model': m, 'contrast': f'{a} - none', 'metric': metric,
                             'mean_diff': float(d.mean()), 'p_wilcoxon_repeat': p, 'n_repeats': len(d)})
    return pd.DataFrame(rows)


def reliability_curve(y, p, n_bins):
    """Equal-frequency bins on the predicted probability -> (mean predicted, observed fraction, count)."""
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    parts = np.array_split(np.arange(len(p)), n_bins)
    return (np.array([p[i].mean() for i in parts]), np.array([y[i].mean() for i in parts]),
            np.array([len(i) for i in parts]))


def make_figure(pred, path, n_bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [m for m in ['LR', 'SVM', 'RF', 'XGBoost', 'LightGBM'] if m in pred['model'].unique()]
    models += [m for m in sorted(pred['model'].unique()) if m not in models]
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
    for ax, m in zip(axs.ravel(), models):
        ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
        for a in ['none', 'smote', 'wgan_gp']:
            g = pred[(pred.model == m) & (pred.augmentation == a)]
            if g.empty:
                continue
            x, yv, _ = reliability_curve(g['y_true'].values, g['prob'].values, n_bins)
            ax.plot(x, yv, 'o-', color=AUG_COLOR[a], label=AUG_LABEL[a], markersize=4, linewidth=1.4)
        ax.set_title(m)
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction of diabetic patients')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    for ax in axs.ravel()[len(models):]:
        ax.axis('off')
    axs.ravel()[0].legend(loc='upper left', fontsize=8, frameon=False)
    fig.suptitle(f'Pooled reliability diagrams of outer-test predictions ({n_bins} equal-frequency bins)')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--outdir', default='calib_v1')
    ap.add_argument('--n-splits', type=int, default=3)
    ap.add_argument('--bins', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pred = pd.read_csv(a.csv)
    need = {'fold', 'augmentation', 'model', 'y_true', 'prob'}
    if need - set(pred.columns):
        raise ValueError(f"Missing columns: {sorted(need - set(pred.columns))}")
    rep = per_repeat_metrics(pred, a.n_splits)
    summ = summarise(rep)
    con = contrasts_vs_none(rep)
    rep.to_csv(os.path.join(a.outdir, 'calibration_per_repeat.csv'), index=False)
    summ.to_csv(os.path.join(a.outdir, 'calibration_summary.csv'), index=False)
    con.to_csv(os.path.join(a.outdir, 'calibration_contrasts_vs_none.csv'), index=False)
    make_figure(pred, os.path.join(a.outdir, 'fig_pooled_reliability.png'), a.bins)
    pd.set_option('display.width', 200)
    print(summ.round(4).to_string(index=False))
    print('\nContrasts vs none (repeat-level, exploratory):')
    print(con.round(4).to_string(index=False))
    print(f"\nWrote results and figure to {a.outdir}/")


if __name__ == '__main__':
    main()
