"""
Fix #9 (v8): a small pytest suite for the correctness-critical helper functions in
imbalanced_v9_perfold_wgan.py -- the ones where a silent bug would be easy to miss (wrong column
indexed, wrong axis, a fold-safety guard that doesn't actually trigger) but would quietly
corrupt every downstream result. This is NOT a full test of the nested-CV pipeline itself
(that would mean re-running the whole expensive benchmark) -- it's meant to catch
regressions when reusing/extending this code for a future study: run this after any change,
before trusting a full multi-hour benchmark run again.

Usage:
    pytest test_imbalanced_v9_perfold_wgan.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imbalanced_v9_perfold_wgan as m


# --- apply_zero_as_missing_imputation ---

def test_zero_imputation_uses_train_median_only():
    X_train = np.array([[0.0, 1.0], [4.0, 1.0], [6.0, 1.0], [0.0, 1.0]])  # col0 median of non-zero {4,6} = 5.0
    X_test = np.array([[0.0, 1.0], [9.0, 1.0]])
    train_out, test_out = m.apply_zero_as_missing_imputation(X_train, X_test, col_indices=[0])
    assert train_out[0, 0] == 5.0
    assert train_out[3, 0] == 5.0
    assert train_out[1, 0] == 4.0  # untouched non-zero value
    assert test_out[0, 0] == 5.0   # test's 0 replaced using TRAIN's median, not test's own
    assert test_out[1, 0] == 9.0   # untouched non-zero value


def test_zero_imputation_ignores_non_target_columns():
    X_train = np.array([[0.0, 0.0], [4.0, 0.0]])
    X_test = np.array([[0.0, 0.0]])
    train_out, test_out = m.apply_zero_as_missing_imputation(X_train, X_test, col_indices=[0])
    assert train_out[0, 1] == 0.0  # column 1 was never in col_indices -- must stay 0
    assert test_out[0, 1] == 0.0


def test_zero_imputation_all_zero_column_is_left_unchanged():
    """If a fold's train column is entirely 0, there's no sensible median to impute with --
    must not crash and must not silently invent a fake median."""
    X_train = np.array([[0.0], [0.0], [0.0]])
    X_test = np.array([[0.0]])
    train_out, test_out = m.apply_zero_as_missing_imputation(X_train, X_test, col_indices=[0])
    assert (train_out[:, 0] == 0.0).all()
    assert test_out[0, 0] == 0.0


# --- apply_smote_augmentation: small-fold safety ---

def test_smote_skips_when_minority_too_small():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(20, 3)), rng.normal(size=(1, 3))])
    y = np.array([0] * 20 + [1])  # minority count = 1 -- SMOTE cannot form any neighbor pair
    X_out, y_out = m.apply_smote_augmentation(X, y)
    assert len(X_out) == len(X)  # returned unchanged, not crashed
    assert len(y_out) == len(y)


def test_smote_reduces_k_neighbors_on_small_minority():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(20, 3)), rng.normal(loc=5, size=(3, 3))])
    y = np.array([0] * 20 + [1] * 3)  # minority count = 3 < default k_neighbors+1=6
    X_out, y_out = m.apply_smote_augmentation(X, y)
    assert (y_out == 0).sum() == (y_out == 1).sum()  # balanced afterwards
    assert len(X_out) > len(X)  # synthetic rows were actually added


# --- _min_class_count / _grid_size ---

def test_min_class_count():
    y = np.array([0, 0, 0, 1, 1])
    assert m._min_class_count(y) == 2


def test_grid_size():
    grid = {'a': [1, 2, 3], 'b': ['x', 'y']}
    assert m._grid_size(grid) == 6


# --- build_tuned_model: falls back instead of crashing on a too-small fold ---

def test_build_tuned_model_falls_back_when_minority_too_small():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.normal(size=(15, 4))
    y = np.array([0] * 14 + [1])  # minority count = 1 -- can't do any stratified CV split
    build_model = lambda seed: LogisticRegression(max_iter=200, random_state=seed)
    model = m.build_tuned_model('LR', build_model, m.CLASSIC_MODEL_PARAM_GRIDS['LR'], X, y, seed=0)
    assert hasattr(model, 'predict_proba')  # got a fitted model back, not an exception
    preds = model.predict(X)
    assert len(preds) == len(y)


# --- assess_synthetic_quality ---

def test_assess_synthetic_quality_flags_out_of_range_values():
    rng = np.random.RandomState(0)
    real = rng.uniform(low=0, high=10, size=(50, 1))
    synthetic_ok = rng.uniform(low=0, high=10, size=(50, 1))       # stays in range
    synthetic_bad = rng.uniform(low=-100, high=-50, size=(50, 1))  # entirely out of range
    synthetic = np.vstack([synthetic_ok, synthetic_bad])
    report = m.assess_synthetic_quality(real, synthetic, feature_names=['x'])
    assert 0.4 < report.loc[0, 'frac_synthetic_out_of_range'] < 0.6


def test_assess_synthetic_quality_ks_stat_low_for_identical_distributions():
    rng = np.random.RandomState(0)
    real = rng.normal(size=(200, 1))
    synthetic = rng.normal(size=(200, 1))  # same distribution, different draw
    report = m.assess_synthetic_quality(real, synthetic, feature_names=['x'])
    assert report.loc[0, 'ks_stat'] < 0.2  # should look similar, not flagged as very different


# --- find_best_threshold ---

def test_find_best_threshold_prefers_balanced_point():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # perfectly separable
    thresh = m.find_best_threshold(y_true, probs)
    preds = (probs >= thresh).astype(int)
    assert (preds == y_true).all()  # a perfectly separable case should find a perfect threshold


# --- apply_wgan_gp_augmentation: smoke test (tiny epochs, just check it runs and shapes are right) ---

def test_wgan_gp_augmentation_smoke_and_shapes():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(30, 4)), rng.normal(loc=3, size=(10, 4))])
    y = np.array([0] * 30 + [1] * 10)
    m.set_seed(0)
    X_aug, y_aug = m.apply_wgan_gp_augmentation(X, y, latent_dim=4, epochs=2, batch_size=8, lr=0.001)
    assert (y_aug == 0).sum() == (y_aug == 1).sum()  # balanced
    assert X_aug.shape[1] == X.shape[1]  # same number of features
    assert len(X_aug) == len(y_aug)


def test_wgan_gp_augmentation_skips_when_already_balanced():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    y = np.array([0] * 10 + [1] * 10)
    X_aug, y_aug = m.apply_wgan_gp_augmentation(X, y, epochs=2, batch_size=4)
    assert len(X_aug) == len(X)  # nothing added -- already balanced


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- v9: per-fold WGAN-GP config selection ---

def _toy_binary_data(n=120, d=4, minority_frac=0.3, seed=0):
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=int)
    y[: int(n * minority_frac)] = 1
    X = rng.randn(n, d) + y[:, None] * 1.5
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_config_label_is_stable_and_handles_none():
    cfg = {'learning_rate': 0.0001, 'epochs': 800, 'batch_size': 64}
    assert m.config_label(cfg) == 'lr0.0001_ep800_bs64'
    assert m.config_label(None) == ''


def test_select_wgan_gp_config_returns_a_candidate_and_balanced_inner_set():
    X, y = _toy_binary_data()
    Xtr, ytr, Xva, yva = X[:90], y[:90], X[90:], y[90:]
    space = [
        {'learning_rate': 0.0002, 'epochs': 5, 'batch_size': 16},
        {'learning_rate': 0.0001, 'epochs': 8, 'batch_size': 16},
    ]
    cfg, Xa, ya, scores = m.select_wgan_gp_config_for_fold(Xtr, ytr, Xva, yva, space, fold_seed=1)
    assert cfg in space
    assert len(scores) == len(space) and all(0.0 <= s <= 1.0 for s in scores)
    # winner's augmented inner-train set must be class-balanced and keep the real rows first
    assert np.bincount(ya)[0] == np.bincount(ya)[1]
    assert np.allclose(Xa[: len(Xtr)], Xtr)


def test_select_wgan_gp_config_never_uses_inner_val_as_training_data():
    X, y = _toy_binary_data()
    Xtr, ytr, Xva, yva = X[:90], y[:90], X[90:], y[90:]
    space = [{'learning_rate': 0.0002, 'epochs': 3, 'batch_size': 16}]
    _, Xa, _, _ = m.select_wgan_gp_config_for_fold(Xtr, ytr, Xva, yva, space, fold_seed=2)
    # the augmented set = real inner-train rows + synthetic rows; no inner-val row appears in it
    real_part = Xa[: len(Xtr)]
    for row in Xva:
        assert not np.any(np.all(np.isclose(real_part, row), axis=1))


def test_run_nested_cv_smoke_records_config_and_qc(monkeypatch):
    X, y = _toy_binary_data(n=90)
    from sklearn.linear_model import LogisticRegression
    builders = {'LR': lambda seed: LogisticRegression(max_iter=200, random_state=seed)}
    space = [
        {'learning_rate': 0.0002, 'epochs': 3, 'batch_size': 16},
        {'learning_rate': 0.0001, 'epochs': 3, 'batch_size': 16},
    ]
    res, qc = m.run_nested_cv(
        X, y, builders, ['none', 'smote', 'wgan_gp'], n_splits=2, n_repeats=1,
        wgan_gp_search_space=space, feature_names=['a', 'b', 'c', 'd'],
        search_n_iter=2, search_cv=2)
    assert set(res['augmentation']) == {'none', 'smote', 'wgan_gp'}
    assert 'pr_auc' in res.columns and res['pr_auc'].between(0, 1).all()
    w = res[res['augmentation'] == 'wgan_gp']
    assert (w['wgan_config'] != '').all()
    assert (res[res['augmentation'] != 'wgan_gp']['wgan_config'] == '').all()
    assert qc is not None and set(qc['fold']) == {1, 2}
    assert set(qc['feature']) == {'a', 'b', 'c', 'd'}
