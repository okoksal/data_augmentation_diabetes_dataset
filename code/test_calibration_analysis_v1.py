import os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import calibration_analysis_v1 as c


def _preds(shift=0.0, n=768, n_repeats=4, n_splits=3, seed=0, aug='none', model='LR'):
    """Outcomes drawn from true probabilities; the model reports logit(p_true)+shift."""
    rng = np.random.RandomState(seed)
    rows = []
    for r in range(n_repeats):
        z = rng.normal(-0.6, 1.2, n)
        y = (rng.uniform(size=n) < c.sigmoid(z)).astype(int)
        p = c.sigmoid(z + shift)
        folds = np.arange(n) % n_splits + 1 + r * n_splits
        rows.append(pd.DataFrame({'fold': folds, 'augmentation': aug, 'model': model,
                                  'row': np.arange(n), 'y_true': y, 'prob': p, 'threshold': 0.4}))
    return pd.concat(rows, ignore_index=True)


def test_perfectly_calibrated_has_slope_one_intercept_zero():
    p = _preds(0.0, n=20000, n_repeats=1)
    y, pr = p['y_true'].values, p['prob'].values
    assert c.calibration_slope(y, pr) == pytest.approx(1.0, abs=0.06)
    assert c.calibration_intercept(y, pr) == pytest.approx(0.0, abs=0.06)


def test_inflated_probabilities_give_negative_intercept_and_positive_gap():
    p = _preds(1.0, n=20000, n_repeats=1)
    y, pr = p['y_true'].values, p['prob'].values
    assert c.calibration_intercept(y, pr) == pytest.approx(-1.0, abs=0.1)
    assert pr.mean() - y.mean() > 0.1


def test_overconfident_model_has_slope_below_one():
    rng = np.random.RandomState(1)
    z = rng.normal(-0.3, 1.0, 20000)
    y = (rng.uniform(size=z.size) < c.sigmoid(z)).astype(int)
    p = c.sigmoid(2.5 * z)                           # too extreme
    assert c.calibration_slope(y, p) == pytest.approx(0.4, abs=0.05)


def test_per_repeat_metrics_pools_folds_within_repeat():
    p = _preds(0.0, n_repeats=4)
    rep = c.per_repeat_metrics(p, 3)
    assert set(rep['repeat']) == {0, 1, 2, 3} and (rep['n'] == 768).all()


def test_reliability_curve_bins_cover_all_samples_and_are_monotone():
    p = _preds(0.0, n=3000, n_repeats=1)
    x, yv, cnt = c.reliability_curve(p['y_true'].values, p['prob'].values, 10)
    assert cnt.sum() == 3000 and np.all(np.diff(x) >= 0)


def test_cli_end_to_end(tmp_path):
    frames = [_preds(s, aug=a, seed=i) for i, (a, s) in enumerate([('none', 0.0), ('smote', 0.8), ('wgan_gp', 0.6)])]
    f = tmp_path / 'p.csv.gz'
    pd.concat(frames).to_csv(f, index=False)
    out = tmp_path / 'o'
    script = os.path.join(os.path.dirname(c.__file__), 'calibration_analysis_v1.py')
    r = subprocess.run([sys.executable, script, str(f), '--outdir', str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name in ['calibration_summary.csv', 'calibration_contrasts_vs_none.csv', 'calibration_per_repeat.csv',
                 'fig_pooled_reliability.png']:
        assert (out / name).exists()
    summ = pd.read_csv(out / 'calibration_summary.csv').set_index('augmentation')
    assert summ.loc['smote', 'intercept'] < summ.loc['none', 'intercept'] - 0.5   # inflated probs detected
