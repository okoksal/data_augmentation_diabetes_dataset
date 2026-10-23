import subprocess, sys, os
import numpy as np
import pandas as pd
import pytest
import nested_cv_statistical_analysis_v5 as s


def test_holm_matches_known_values_and_is_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = s.holm(p)
    # sorted p: .005,.01,.03,.04 -> *4,*3,*2,*1 = .02,.03,.06,.04 -> running max .02,.03,.06,.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.all(adj >= p) and np.all(adj <= 1)


def test_holm_caps_at_one():
    assert np.allclose(s.holm([0.5, 0.9]), [1.0, 1.0])


def test_nadeau_bengio_inflates_variance_vs_plain_t():
    rng = np.random.RandomState(0)
    diff = rng.normal(0.01, 0.02, 45)
    se, p = s.nadeau_bengio(diff, 0.5)
    plain_se = diff.std(ddof=1) / np.sqrt(len(diff))
    assert se > plain_se * 3          # (1/45 + 0.5) vs (1/45): ~sqrt(23.5) = 4.8x
    assert 0 <= p <= 1


def test_safe_wilcoxon_handles_all_zero_differences():
    assert s.safe_wilcoxon(np.zeros(10)) == 1.0


def _fake_results(n_repeats=6, n_splits=3, effect=0.05, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for fold in range(1, n_repeats * n_splits + 1):
        base = rng.normal(0, 0.01)
        for model in ['LR', 'RF', 'SVM']:
            for aug in ['none', 'smote', 'wgan_gp']:
                shift = effect if (model == 'RF' and aug == 'wgan_gp') else 0.0
                f1 = 0.65 + base + shift + rng.normal(0, 0.005)
                row = dict(fold=fold, augmentation=aug, model=model, threshold=0.4, collapsed=False)
                for k in s.METRICS:
                    row[k] = f1 if k == 'f1' else rng.uniform(0.5, 0.9)
                rows.append(row)
    return pd.DataFrame(rows)


def test_load_assigns_repeat_ids(tmp_path):
    p = tmp_path / 'r.csv'
    _fake_results().to_csv(p, index=False)
    d = s.load(str(p), 3)
    assert d.groupby('fold')['repeat'].first().tolist() == [i // 3 for i in range(18)]


def test_load_rejects_missing_columns(tmp_path):
    p = tmp_path / 'r.csv'
    _fake_results().drop(columns=['pr_auc']).to_csv(p, index=False)
    with pytest.raises(ValueError):
        s.load(str(p), 3)


def test_contrasts_detect_planted_effect_and_not_null_effects(tmp_path):
    p = tmp_path / 'r.csv'
    _fake_results(n_repeats=15).to_csv(p, index=False)
    d = s.load(str(p), 3)
    c = s.augmentation_contrasts(d, 0.5, np.random.RandomState(0), 2000)
    f1 = c[c.metric == 'f1'].set_index(['model', 'contrast'])
    planted = f1.loc[('RF', 'wgan_gp - none')]
    assert planted.mean_diff == pytest.approx(0.05, abs=0.01)
    assert planted.p_holm_fold < 0.001 and planted.p_holm_repeat < 0.01
    assert planted.ci_repeat_lo > 0
    assert f1.loc[('LR', 'wgan_gp - none')].p_holm_fold > 0.05    # null effect stays non-significant
    assert len(f1) == 9                                            # 3 models x 3 pairs
    assert c.groupby('metric').size().nunique() == 1


def test_marginal_and_model_tests_shapes(tmp_path):
    p = tmp_path / 'r.csv'
    _fake_results().to_csv(p, index=False)
    d = s.load(str(p), 3)
    assert set(s.marginal_tests(d)['metric']) == set(s.METRICS)
    assert set(s.model_tests(d)['augmentation']) == {'none', 'smote', 'wgan_gp'}
    top = s.top_combo_contrasts(d, 0.5, np.random.RandomState(0), 500)
    assert top['best'].iloc[0] == 'RF+wgan_gp'                      # the planted winner
    assert len(top) == 8                                           # 9 combos - the best one


def test_cli_end_to_end_writes_all_outputs(tmp_path):
    p = tmp_path / 'r.csv'
    _fake_results().to_csv(p, index=False)
    out = tmp_path / 'out'
    script = os.path.join(os.path.dirname(s.__file__), 'nested_cv_statistical_analysis_v5.py')
    r = subprocess.run([sys.executable, script, str(p), '--outdir', str(out), '--n-boot', '300'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ['report.txt', 'table4_cell_summary.csv', 'augmentation_contrasts_all_metrics.csv',
              'marginal_augmentation_tests.csv', 'friedman_models_within_augmentation.csv',
              'best_f1_combo_vs_others.csv']:
        assert (out / f).exists()
