import os

# Fix: joblib worker subprocesses (spawned for n_jobs=-1 in RandomForest, SMOTE's internal
# nearest-neighbor search, RandomizedSearchCV, etc.) run in their own separate Python
# interpreter and do NOT inherit the `warnings.filterwarnings("ignore")` call below -- that
# only affects the main process. Each worker reads PYTHONWARNINGS from its environment at
# startup, so setting it here (before joblib/sklearn are imported and before any worker pool
# is spawned) silences the same warnings in the workers too, e.g. the harmless
# "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
# UserWarning some imbalanced-learn/scikit-learn version combinations produce.
os.environ["PYTHONWARNINGS"] = "ignore"

import random
import warnings
import time
import sys
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Fix #3: non-interactive backend -- plots are saved straight to file and never popped up
# as a window, so there's nothing to close by hand. Must be set before pyplot is imported.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, StratifiedKFold, ParameterSampler
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# Fix #8 (v8): nested-CV fold counts, now overridable from the command line
# (--n-splits / --n-repeats) instead of being hardcoded at every call site.
N_SPLITS = 3
N_REPEATS = 15

# Fix #1: use a GPU for WGAN-GP training when one is available -- with 270 separate
# WGAN-GP trainings across the full benchmark (45 folds x 2 inner/outer x 3 datasets),
# this is the single biggest runtime lever in the script.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HYPERPARAMETER SEARCH SPACES ---
# Two different strategies, on purpose (see the discussion this replaces a manual
# "try lr=0.0001, then try lr=0.0002 by hand" workflow):
#
# 1) Classical models (cheap to train) -> REAL automated search, run independently inside
#    every inner-train / outer-train fold via RandomizedSearchCV. Its own CV folds are
#    carved entirely out of whichever training data is passed in -- outer-test and
#    inner-val are never touched by the search, so this is genuinely nested / leakage-free.
CLASSIC_MODEL_PARAM_GRIDS = {
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31],
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 6],
    },
    'RF': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
    },
    'SVM': {
        # CalibratedClassifierCV(estimator=SVC(...)) -> params need the 'estimator__' prefix.
        'estimator__C': [0.1, 1, 10],
        'estimator__gamma': ['scale', 'auto'],
    },
    'LR': {
        'C': [0.01, 0.1, 1, 10],
    },
}
N_SEARCH_ITER = 5      # RandomizedSearchCV candidates tried per fold/model/augmentation
SEARCH_CV_FOLDS = 3    # only used by the illustrative single-split plots (build_tuned_model)
INNER_FOLDS = 5        # inner CV folds inside every outer-train: config/hyperparameter search AND threshold
SAVE_PREDICTIONS = False  # --save-predictions: also write every outer-test probability to a CSV

# 2) WGAN-GP (expensive to train) -> selected PER OUTER FOLD from the 4 candidate configs below,
#    by an inner K-fold CV on that fold's outer-train set (see select_wgan_gp_config_inner_cv()).
#    v8 chose one config once per dataset on a split of the FULL dataset (outside the nested
#    CV), so the choice was informed by labels of samples that later landed in outer test
#    folds. Selecting inside every outer fold removes that: no outer-test sample ever
#    influences any selection.
WGAN_GP_SEARCH_SPACE = [
    {'learning_rate': 0.0001, 'epochs': 500, 'batch_size': 32},
    {'learning_rate': 0.0002, 'epochs': 500, 'batch_size': 32},
    {'learning_rate': 0.0002, 'epochs': 800, 'batch_size': 64},
    {'learning_rate': 0.0001, 'epochs': 800, 'batch_size': 64},
]

# Dataset CSVs (diabetes.csv/heart.csv/breast_cancer.csv) are cached here, next to the
# script itself -- NOT inside a per-run folder -- so they are downloaded once and reused
# across every future run regardless of CLI options. This is what fixes the classic "it ran
# with no errors but I can't find the files" problem: some IDEs (Spyder, certain Jupyter/
# VS Code configs) execute scripts from a working directory that is silently different from
# the folder shown in your file explorer. Anchoring to the script's own path removes that
# ambiguity entirely.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Fix #7 (v8): every RUN's own outputs (result CSVs, plots, run manifest, log) go into a
# fresh timestamped subfolder instead of overwriting the previous run's files. Set once at
# startup in the __main__ block (after CLI args are parsed) -- left as None until then so
# any accidental early use fails loudly instead of silently writing to the wrong place.
RUN_DIR = None

# Fix #5 (v8): collects per-dataset facts (chosen WGAN-GP config, QC summary, best combo,
# elapsed time) as each dataset finishes, so write_run_manifest() can assemble the final
# run_config.md without re-deriving anything -- each dataset just reports what it did.
RUN_MANIFEST_INFO = {}


def data_path(filename):
    """Absolute path for a cached dataset CSV inside the stable DATA_DIR (not per-run)."""
    return os.path.join(DATA_DIR, filename)


def out(filename):
    """Builds an absolute save path inside this run's own RUN_DIR for the given filename."""
    if RUN_DIR is None:
        raise RuntimeError("RUN_DIR is not set yet -- out() called before the __main__ "
                            "startup block initialized the run folder.")
    return os.path.join(RUN_DIR, filename)


class _Tee:
    """
    Fix #10 (v8): mirrors everything written to stdout/stderr into a run.log file as well,
    so the full console transcript survives even if the terminal is closed/lost during a
    long run, and can be attached as supplementary material alongside the paper. This
    captures every existing print() call automatically -- no need to convert them to
    logging.info() one by one.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_logging(run_dir):
    """Redirects stdout/stderr to also write into '<run_dir>/run.log'. Returns the log path."""
    log_path = os.path.join(run_dir, 'run.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def save_dataset_copy(df, file_path):
    """
    Saves a freshly-downloaded dataset to file_path (inside DATA_DIR) so the next run
    finds it locally and does not need to re-download it. Best-effort: a failure here
    (e.g. read-only folder) is only a warning, never fatal -- the run continues either way
    since the data itself is already loaded in memory.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"[OK] Cached a local copy of the downloaded dataset at '{file_path}' "
              f"(future runs will load it from disk instead of re-downloading).")
    except OSError as e:
        print(f"[WARN] Could not cache the downloaded dataset to '{file_path}' ({e}). "
              f"This run continues normally, but it will be re-downloaded next time.")


def set_seed(seed):
    """Seed every RNG involved (numpy, python random, torch) for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 1. DATA LOADING AND PREPROCESSING (PIMA INDIANS) ---
def load_and_preprocess_pima(file_path='diabetes.csv'):
    """
    Loads and prepares the classic 768-row Pima Indians Diabetes Dataset.
    If the file is not found locally (looked up in DATA_DIR, i.e. next to this script),
    it automatically downloads it from the web repository and saves a local copy in
    DATA_DIR, so subsequent runs load it from disk instead of re-downloading.
    """
    if not os.path.isabs(file_path):
        file_path = data_path(file_path)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[INFO] {file_path} not found locally. Fetching from remote repository...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                   'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        save_dataset_copy(df, file_path)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print(f"[OK] Pima Dataset Loaded: {len(df)} rows")
    print(f"[INFO] Original Class Distribution: {np.bincount(y)[0]} Non-Diabetic vs {np.bincount(y)[1]} Diabetic")
    return X, y


# --- 1b. DATA LOADING AND PREPROCESSING (KAGGLE HEART DISEASE) ---
def load_and_preprocess_heart(file_path='heart.csv'):
    """
    Loads and prepares the Kaggle Heart Disease (UCI Cleveland) dataset -- 303 rows,
    14 columns, binary 'target' (1 = presence of heart disease).
    If the file is not found locally (looked up in DATA_DIR, i.e. next to this script;
    e.g. downloaded from Kaggle as 'heart.csv'), it automatically downloads an equivalent
    mirror of the same dataset from GitHub and saves a local copy in DATA_DIR, so
    subsequent runs load it from disk instead of re-downloading.
    """
    if not os.path.isabs(file_path):
        file_path = data_path(file_path)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[INFO] {file_path} not found locally. Fetching from remote repository...")
        url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
        df = pd.read_csv(url)
        save_dataset_copy(df, file_path)

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"[OK] Heart Disease Dataset Loaded: {len(df)} rows")
    print(f"[INFO] Original Class Distribution: {np.bincount(y)[0]} No-Disease vs {np.bincount(y)[1]} Disease")
    return X, y


# --- 1c. DATA LOADING AND PREPROCESSING (KAGGLE BREAST CANCER WISCONSIN) ---
def load_and_preprocess_breast_cancer(file_path='breast_cancer.csv'):
    """
    Loads and prepares the Kaggle Breast Cancer Wisconsin (Diagnostic) dataset -- 569 rows,
    30 numeric features, binary diagnosis (Malignant = 1, Benign = 0).
    If the file is not found locally (looked up in DATA_DIR, i.e. next to this script;
    e.g. downloaded from Kaggle as 'breast_cancer.csv', with an 'id' column and an M/B
    'diagnosis' column), it falls back to the identical Wisconsin Diagnostic Breast Cancer
    data bundled with scikit-learn so the script still runs offline, and saves a local copy
    (in the same 'diagnosis' M/B format) to DATA_DIR so subsequent runs load it from disk
    instead of rebuilding it. The scikit-learn target is flipped (1 - target) so that,
    consistently with the other two datasets, 1 always denotes the clinically
    positive/concerning class.
    """
    if not os.path.isabs(file_path):
        file_path = data_path(file_path)

    try:
        df = pd.read_csv(file_path)
        drop_cols = [c for c in df.columns if c.lower() in ('id', 'unnamed: 32')]
        df = df.drop(columns=drop_cols, errors='ignore')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
        X = df.drop('diagnosis', axis=1)
    except FileNotFoundError:
        print(f"[INFO] {file_path} not found locally. Falling back to the equivalent "
              f"scikit-learn 'Breast Cancer Wisconsin (Diagnostic)' bundled dataset...")
        data = load_breast_cancer(as_frame=True)
        X = data.frame.drop('target', axis=1)
        y = 1 - data.frame['target']  # sklearn: 0=malignant,1=benign -> flip to 1=malignant

        df_to_cache = X.copy()
        df_to_cache['diagnosis'] = y.map({1: 'M', 0: 'B'})
        save_dataset_copy(df_to_cache, file_path)

    print(f"[OK] Breast Cancer Dataset Loaded: {len(X)} rows")
    print(f"[INFO] Original Class Distribution: {np.bincount(y)[0]} Benign vs {np.bincount(y)[1]} Malignant")
    return X, y


# --- 1d. OPTIONAL PREPROCESSING: ZERO-AS-MISSING MEDIAN IMPUTATION ---
# Toggle: turns the whole preprocessing option on/off without touching the per-dataset config.
APPLY_ZERO_AS_MISSING_IMPUTATION = True

# Only Pima Indians Diabetes gets this treatment, and only for columns where 0 is
# biologically impossible (you can't have 0 glucose, 0 blood pressure, 0 BMI, etc.) and
# is almost certainly the original data collection's missing-value sentinel. Deliberately
# left empty for Heart and Breast Cancer: in those datasets a 0 can be a genuine value
# (e.g. a categorical flag, or "symptom absent"), so blindly imputing every 0 there would
# corrupt real data instead of fixing missing data. See each dataset's entry in DATASETS.
PIMA_ZERO_AS_MISSING_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def apply_zero_as_missing_imputation(X_train, X_ref, col_indices):
    """
    For the given column indices, treats 0 as a missing-value sentinel and replaces it with
    that column's TRAIN-ONLY median (computed after excluding the zeros themselves, i.e. the
    median of the genuine non-zero measurements). The median is always computed from X_train
    and then applied to both X_train and X_ref -- X_ref may be X_train itself (nothing wrong
    with that) or a separate val/test split, but it never contributes to computing the
    median. This keeps imputation leakage-free, the same way scaling/SMOTE/WGAN-GP are
    already handled fold-by-fold elsewhere in this script. Must run on the RAW (unscaled)
    features, since scaling would shift the 0 sentinel to some other number.
    """
    X_train = X_train.copy()
    X_ref = X_ref.copy()
    for col in col_indices:
        train_col = X_train[:, col]
        non_zero_vals = train_col[train_col != 0]
        if non_zero_vals.size == 0:
            continue  # every value in this fold's train column is 0 -- nothing sensible to impute with
        median_val = np.median(non_zero_vals)
        X_train[X_train[:, col] == 0, col] = median_val
        X_ref[X_ref[:, col] == 0, col] = median_val
    return X_train, X_ref


# --- 2. GENERATIVE MODEL: WGAN-GP (WASSERSTEIN GAN WITH GRADIENT PENALTY) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim)
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, data_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def apply_wgan_gp_augmentation(X_train, y_train, latent_dim=16, epochs=800, batch_size=64,
                                lr=0.0002, verbose=False):
    """
    Trains a WGAN-GP solely on the minority (diabetic) class of the TRAINING split it is given,
    then generates enough synthetic samples to reach a balanced 1:1 class ratio.

    IMPORTANT: caller is responsible for only ever passing data that does not touch the
    outer/inner evaluation folds, otherwise synthetic samples correlated with held-out
    points would leak information into the evaluation.
    """
    # Fix #4: don't assume class 0 is always the majority -- derive minority/majority
    # from the actual counts so this still works correctly if it's ever reused on a
    # dataset where the positive class (1) happens to be the majority.
    counts = np.bincount(y_train, minlength=2)
    minority_label = int(np.argmin(counts))
    majority_label = int(np.argmax(counts))
    minority_data = X_train[y_train == minority_label]
    majority_count = int(counts[majority_label])
    minority_count = int(counts[minority_label])
    n_needed = abs(majority_count - minority_count)

    if n_needed == 0 or minority_count < 2:
        # Already balanced (or too few minority samples to model at all) -- skip GAN training.
        return X_train, y_train

    # Fix #1: move everything onto DEVICE (GPU if available) for training.
    minority_tensor = torch.FloatTensor(minority_data).to(DEVICE)
    data_dim = X_train.shape[1]
    generator = Generator(latent_dim, data_dim).to(DEVICE)
    critic = Critic(data_dim).to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    fold_batch_size = min(batch_size, len(minority_tensor))

    if verbose:
        print(f"    -> WGAN-GP training on {minority_count} minority samples ({epochs} epochs)...")

    for epoch in range(epochs):
        for _ in range(5):  # 5 critic steps per generator step (WGAN standard)
            idx = np.random.choice(len(minority_tensor), fold_batch_size, replace=True)
            real_samples = minority_tensor[idx]

            z = torch.randn(real_samples.size(0), latent_dim, device=DEVICE)
            fake_samples = generator(z).detach()

            loss_C = -torch.mean(critic(real_samples)) + torch.mean(critic(fake_samples))
            gp = compute_gradient_penalty(critic, real_samples, fake_samples)
            loss_C += 10 * gp

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        z = torch.randn(fold_batch_size, latent_dim, device=DEVICE)
        fake_samples = generator(z)
        loss_G = -torch.mean(critic(fake_samples))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if verbose and (epoch + 1) % 200 == 0:
            print(f"       Epoch [{epoch + 1}/{epochs}] | Loss C: {loss_C.item():.4f} | Loss G: {loss_G.item():.4f}")

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_needed, latent_dim, device=DEVICE)
        synthetic_samples = generator(z).cpu().numpy()

    X_resampled = np.vstack([X_train, synthetic_samples])
    y_resampled = np.concatenate([y_train, np.full(n_needed, minority_label)]).astype(int)
    X_resampled = np.ascontiguousarray(X_resampled)

    return X_resampled, y_resampled


def apply_smote_augmentation(X_train, y_train, random_state=RANDOM_SEED):
    """
    Fix #3: SMOTE's default k_neighbors=5 requires at least 6 minority samples
    (it needs k neighbors *other than* the point itself) and raises otherwise --
    a real risk here since nested CV keeps carving the already-small minority class
    into smaller inner folds (e.g. the 303-row Heart dataset). We shrink k_neighbors
    to fit the fold, and skip augmentation entirely (returning the data unchanged,
    same as the 'none' condition) if there are too few minority samples to form even
    a single neighbor pair.
    """
    minority_count = int(np.bincount(y_train).min())
    if minority_count < 2:
        print(f"[WARN] SMOTE skipped: minority class has only {minority_count} sample(s) "
              f"in this fold, not enough to find any neighbor. Returning data unaugmented.")
        return X_train, y_train

    k_neighbors = min(5, minority_count - 1)
    if k_neighbors < 5:
        print(f"[WARN] SMOTE: minority class has only {minority_count} samples in this fold "
              f"-- reducing k_neighbors from 5 to {k_neighbors}.")

    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return sm.fit_resample(X_train, y_train)


def apply_augmentation(name, X_train, y_train, seed, wgan_gp_config=None):
    """
    Dispatch to the requested augmentation strategy. 'none' returns the data untouched.
    wgan_gp_config, if given, is a dict with the WGAN-GP settings to use. The nested CV does NOT
    use this path for 'wgan_gp' (it selects the config per fold via
    select_wgan_gp_config_inner_cv()); it is only used for the illustrative single-split plots.
    """
    if name == 'none':
        return X_train, y_train
    elif name == 'smote':
        return apply_smote_augmentation(X_train, y_train, random_state=seed)
    elif name == 'wgan_gp':
        config = wgan_gp_config or {}
        return apply_wgan_gp_augmentation(
            X_train, y_train,
            latent_dim=config.get('latent_dim', 16),
            epochs=config.get('epochs', 800),
            batch_size=config.get('batch_size', 64),
            lr=config.get('learning_rate', 0.0002),
        )
    else:
        raise ValueError(f"Unknown augmentation strategy: {name}")


# --- 2b. INNER-CV HELPERS (WGAN-GP config selection) ---
def config_label(config):
    """Compact, CSV-friendly label for a WGAN-GP config, e.g. 'lr0.0001_ep800_bs64'."""
    if config is None:
        return ''
    return f"lr{config['learning_rate']}_ep{config['epochs']}_bs{config['batch_size']}"


def _min_class_count(y):
    counts = np.bincount(np.asarray(y).astype(int))
    counts = counts[counts > 0]
    return int(counts.min()) if counts.size else 0


def make_inner_splits(X, y, inner_folds, seed):
    """Stratified inner K-fold split of the outer-train set -> list of (train_idx, val_idx)."""
    k = min(inner_folds, _min_class_count(y))
    if k < 2:
        raise ValueError("Outer-train set has too few minority samples for an inner CV.")
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))


def select_wgan_gp_config_inner_cv(X_T, y_T, splits, search_space, fold_seed):
    """
    Selects the WGAN-GP config (learning_rate / epochs / batch_size) for ONE outer fold with
    an inner K-fold CV over that fold's outer-train set T (outer-test never enters).

    For every candidate and every inner split k: train a WGAN-GP on the inner-train part
    only, balance it, fit a fixed default LightGBM, predict the (real, un-augmented)
    inner-val part. The candidate's score is the PR-AUC (average precision) of the pooled
    out-of-fold predictions -- threshold-free, and computed on real samples only.

    The winner's augmented inner-train sets are returned so they are reused (not retrained)
    for the threshold/hyperparameter stage. GAN trainings: len(search_space) x len(splits).

    Returns (best_config, aug_sets_of_best, candidate_scores).
    """
    best = None
    scores = []
    for config in search_space:
        sets = []
        oof = np.zeros(len(y_T))
        for k, (tr, va) in enumerate(splits):
            set_seed(fold_seed * 100 + k)  # same seed per split for every candidate -> fair comparison
            Xa, ya = apply_wgan_gp_augmentation(
                X_T[tr], y_T[tr],
                latent_dim=config.get('latent_dim', 16),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                lr=config['learning_rate'],
            )
            scorer = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                         random_state=fold_seed, verbosity=-1)
            scorer.fit(Xa, ya)
            oof[va] = scorer.predict_proba(X_T[va])[:, 1]
            sets.append((Xa, ya))
        score = float(average_precision_score(y_T, oof))
        scores.append(score)
        if best is None or score > best[0]:  # strict '>' -> ties go to the earliest candidate
            best = (score, config, sets)
    return best[1], best[2], scores


def assess_synthetic_quality(real_values, synthetic_values, feature_names):
    """
    Fix #3 (v8): sanity-checks how plausible the WGAN-GP-generated synthetic minority
    samples are, feature by feature, in the ORIGINAL (unscaled) units -- this is what a
    reviewer means by "did you check the synthetic data is realistic?" for a GAN-based
    augmentation method. Two simple, standard checks per feature (no external deps beyond
    scipy, already a dependency here):
      - Kolmogorov-Smirnov two-sample test comparing the real vs synthetic marginal
        distributions (larger ks_stat / smaller ks_p_value => the two distributions differ
        more).
      - Fraction of synthetic values that fall outside the observed real min/max range for
        that feature (e.g. a synthetic negative Age or negative Glucose would show up here).
    This is a reporting/QC step only -- it does not filter or reject any synthetic sample,
    it just gives citable evidence (a table, in the paper's appendix or supplementary
    material) that the augmented data was checked rather than assumed plausible.
    """
    real_values = np.asarray(real_values)
    synthetic_values = np.asarray(synthetic_values)
    rows = []
    for i, name in enumerate(feature_names):
        real_col = real_values[:, i]
        syn_col = synthetic_values[:, i]
        ks_stat, ks_p = scipy_stats.ks_2samp(real_col, syn_col)
        real_min, real_max = float(real_col.min()), float(real_col.max())
        out_of_range = float(np.mean((syn_col < real_min) | (syn_col > real_max)))
        rows.append({
            'feature': name, 'ks_stat': ks_stat, 'ks_p_value': ks_p,
            'real_min': real_min, 'real_max': real_max,
            'frac_synthetic_out_of_range': out_of_range,
        })
    return pd.DataFrame(rows).sort_values('ks_stat', ascending=False)


# --- 4. THRESHOLD SELECTION ---
def find_best_threshold(y_true, probs):
    """Threshold that maximizes F1 on the given (validation) set only."""
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    f1_vals = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_vals)
    return thresh[best_idx] if best_idx < len(thresh) else 0.5


# --- 5. AUTOMATED INNER-FOLD HYPERPARAMETER SEARCH (classical models only) ---
def _grid_size(param_grid):
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def build_tuned_model(model_name, build_model, param_grid, X, y, seed,
                       n_iter=N_SEARCH_ITER, cv=SEARCH_CV_FOLDS):
    """
    Replaces manual hand-tuning for the classical models with a real, automated search:
    RandomizedSearchCV is fit on X/y ONLY (whatever training split is handed to this
    function -- inner-train or outer-train), using its own internal CV folds carved out of
    that same data. The outer-test / inner-val splits used for scoring elsewhere never enter
    this function, so results stay leakage-free and the "searched within inner folds" claim
    in a methods section is literally true here.

    Falls back to the fixed default hyperparameters (no search) if a fold's minority class
    is too small to support stratified CV at all -- this can happen in the smaller datasets
    (e.g. Heart) once nested CV has carved the data down repeatedly.
    """
    safe_cv = min(cv, _min_class_count(y))
    if safe_cv < 2:
        model = build_model(seed)
        model.fit(X, y)
        return model

    search = RandomizedSearchCV(
        estimator=build_model(seed),
        param_distributions=param_grid,
        n_iter=min(n_iter, _grid_size(param_grid)),
        scoring='f1',
        cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,  # avoid oversubscription with RandomForest/LightGBM's own internal threading
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


# --- 6. REPEATED NESTED CROSS-VALIDATION ---
def tune_and_threshold(build_model, param_grid, aug_sets, X_T, y_T, splits, n_iter, seed):
    """
    Hyperparameter search AND decision-threshold selection from ONE shared set of
    out-of-fold (OOF) predictions on the outer-train set T.

    aug_sets[k] is the already-augmented INNER-TRAIN set of inner split k (augmentation was
    fitted on that inner-train part only). Every candidate hyperparameter set is fitted on
    each aug_sets[k] and predicts the REAL inner-val rows of split k, so the search never
    scores on synthetic samples and never sees augmentation built from its own validation
    rows. A candidate's score = PR-AUC of its pooled OOF predictions. The F1-optimal
    threshold is then taken from the winning candidate's pooled OOF predictions
    (len(T) samples, not a single 25% hold-out).

    Returns (best_params, oof_probs_of_best, threshold, candidate_scores).
    """
    n = min(n_iter, _grid_size(param_grid))
    candidates = list(ParameterSampler(param_grid, n_iter=n, random_state=seed))
    best = None
    scores = []
    for params in candidates:
        oof = np.zeros(len(y_T))
        for (tr, va), (Xa, ya) in zip(splits, aug_sets):
            m = clone(build_model(seed))
            m.set_params(**params)
            m.fit(Xa, ya)
            oof[va] = m.predict_proba(X_T[va])[:, 1]
        score = float(average_precision_score(y_T, oof))
        scores.append(score)
        if best is None or score > best[0]:
            best = (score, params, oof)
    threshold = find_best_threshold(y_T, best[2])
    return best[1], best[2], threshold, scores


def run_nested_cv(X, y, model_builders, augmentations, n_splits=3, n_repeats=15,
                   inner_folds=None, base_seed=RANDOM_SEED, wgan_gp_search_space=None,
                   impute_zero_col_indices=None, feature_names=None, search_n_iter=None,
                   prediction_sink=None):
    """
    prediction_sink (optional list): if given, one DataFrame per fitted cell is appended with the
    outer-test predictions (fold, augmentation, model, original row index, y_true, prob, threshold).
    This only records values that are computed anyway; results are unchanged.

    Outer loop (n_splits x n_repeats folds): the held-out test fold E is used ONLY for the
    final scoring of one fitted model per (augmentation, classifier) cell.

    Inside every outer fold, on the outer-train set T only, an INNER stratified K-fold CV
    (K = inner_folds, default INNER_FOLDS) does, for every augmentation condition:
      1. augmentation fitted on each inner-train part only (SMOTE / WGAN-GP) -- the inner-val
         part is always real data;
      2. WGAN-GP config selection (4 candidates; pooled-OOF PR-AUC of a fixed LightGBM);
      3. classifier hyperparameter search (RandomizedSearchCV-style, n_iter candidates;
         pooled-OOF PR-AUC) -- replaces the old search whose CV folds contained synthetic
         samples;
      4. decision threshold = F1-optimal threshold on the winning candidate's pooled OOF
         predictions over all of T.
    Then the final model (winning hyperparameters) is fitted on the augmented T and scored
    on E at that threshold. Because threshold and final model share the same hyperparameters
    and the same pipeline, the inner-model/outer-model probability-scale mismatch that made
    v8/v9 thresholds collapse in a few folds is largely removed; the remaining difference is
    that the final model's GAN is a different random draw than the inner ones.

    GAN trainings per outer fold = len(WGAN search space) x K + 1.

    impute_zero_col_indices (optional, Pima only): zero-as-missing median imputation fitted on
    T only, applied to T and E, BEFORE scaling.

    Returns (results_df, qc_df). qc_df is None if no synthetic samples were generated.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    inner_folds = INNER_FOLDS if inner_folds is None else inner_folds
    search_n_iter = N_SEARCH_ITER if search_n_iter is None else search_n_iter
    wgan_gp_search_space = wgan_gp_search_space or WGAN_GP_SEARCH_SPACE
    cols = list(feature_names) if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=base_seed)
    total_folds = n_splits * n_repeats
    records = []
    qc_frames = []
    fold_id = 0

    for outer_train_idx, outer_test_idx in rskf.split(X, y):
        fold_id += 1
        fold_start = time.time()
        fold_seed = base_seed + fold_id
        set_seed(fold_seed)

        X_T, X_E = X[outer_train_idx], X[outer_test_idx]
        y_T, y_E = y[outer_train_idx], y[outer_test_idx]

        if impute_zero_col_indices:
            X_T, X_E = apply_zero_as_missing_imputation(X_T, X_E, impute_zero_col_indices)

        scaler = StandardScaler()
        X_T_s = scaler.fit_transform(X_T)
        X_E_s = scaler.transform(X_E)

        splits = make_inner_splits(X_T_s, y_T, inner_folds, fold_seed)

        # --- augmentation sets: per inner split (inner-train only) and final (all of T) ---
        aug_inner, aug_final, chosen_config = {}, {}, None
        for aug_name in augmentations:
            if aug_name == 'wgan_gp':
                chosen_config, aug_inner[aug_name], cand_scores = select_wgan_gp_config_inner_cv(
                    X_T_s, y_T, splits, wgan_gp_search_space, fold_seed)
                print(f"[Fold {fold_id:>2}/{total_folds}] WGAN-GP config -> {config_label(chosen_config)} "
                      f"(pooled-OOF PR-AUC per candidate: {[round(s, 3) for s in cand_scores]})")
                set_seed(fold_seed)
                aug_final[aug_name] = apply_wgan_gp_augmentation(
                    X_T_s, y_T,
                    latent_dim=chosen_config.get('latent_dim', 16),
                    epochs=chosen_config['epochs'],
                    batch_size=chosen_config['batch_size'],
                    lr=chosen_config['learning_rate'],
                )
                Xf = aug_final[aug_name][0]
                n_syn = len(Xf) - len(X_T_s)
                if n_syn > 0:
                    minority_label = int(np.argmin(np.bincount(y_T)))
                    real_raw = scaler.inverse_transform(X_T_s[y_T == minority_label])
                    syn_raw = scaler.inverse_transform(Xf[len(X_T_s):])
                    qc_fold = assess_synthetic_quality(real_raw, syn_raw, cols)
                    qc_fold.insert(0, 'fold', fold_id)
                    qc_frames.append(qc_fold)
            else:
                aug_inner[aug_name] = [
                    apply_augmentation(aug_name, X_T_s[tr], y_T[tr], fold_seed + k)
                    for k, (tr, va) in enumerate(splits)]
                aug_final[aug_name] = apply_augmentation(aug_name, X_T_s, y_T, fold_seed)

        for aug_name in augmentations:
            wcfg = config_label(chosen_config) if aug_name == 'wgan_gp' else ''
            for model_name, build_model in model_builders.items():
                best_params, oof, thr, _ = tune_and_threshold(
                    build_model, CLASSIC_MODEL_PARAM_GRIDS[model_name], aug_inner[aug_name],
                    X_T_s, y_T, splits, search_n_iter, fold_seed)

                final_model = clone(build_model(fold_seed))
                final_model.set_params(**best_params)
                final_model.fit(*aug_final[aug_name])
                test_probs = final_model.predict_proba(X_E_s)[:, 1]
                preds = (test_probs >= thr).astype(int)
                if prediction_sink is not None:
                    prediction_sink.append(pd.DataFrame({
                        'fold': fold_id, 'augmentation': aug_name, 'model': model_name,
                        'row': outer_test_idx, 'y_true': y_E, 'prob': test_probs, 'threshold': thr}))

                cm = confusion_matrix(y_E, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                test_pos_rate = float(preds.mean())
                oof_pos_rate = float((oof >= thr).mean())

                record = {
                    'fold': fold_id,
                    'augmentation': aug_name,
                    'model': model_name,
                    'wgan_config': wcfg,
                    'best_params': str(best_params),
                    'inner_folds': len(splits),
                    'threshold': thr,
                    'f1': f1_score(y_E, preds, zero_division=0),
                    'recall': recall_score(y_E, preds, zero_division=0),
                    'auc_roc': roc_auc_score(y_E, test_probs),
                    'pr_auc': average_precision_score(y_E, test_probs),
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'false_negatives': fn,
                    'brier_score': brier_score_loss(y_E, test_probs),
                    # diagnostics for threshold transfer: OOF-side vs test-side behaviour
                    'oof_f1': f1_score(y_T, (oof >= thr).astype(int), zero_division=0),
                    'oof_pos_rate': oof_pos_rate,
                    'test_pos_rate': test_pos_rate,
                    'collapsed': bool(test_pos_rate > 0.97 or test_pos_rate < 0.03),
                }
                records.append(record)
                print(f"[Fold {fold_id:>2}/{total_folds}] {aug_name:8s} | {model_name:8s} | "
                      f"thr={thr:.3f} | F1={record['f1']:.3f} | AUC={record['auc_roc']:.3f} | "
                      f"PR-AUC={record['pr_auc']:.3f} | Brier={record['brier_score']:.3f} | FN={fn} | "
                      f"pos-rate OOF/test={oof_pos_rate:.2f}/{test_pos_rate:.2f}")
        print(f"[Fold {fold_id:>2}/{total_folds}] done in {(time.time() - fold_start) / 60:.1f} min")

    qc_df = pd.concat(qc_frames, ignore_index=True) if qc_frames else None
    return pd.DataFrame(records), qc_df


# --- 7. ANALYSIS AND VISUALIZATION UTILITIES ---
def plot_clinical_importance(model, feature_names, model_name, dataset_display_name, output_prefix):
    """
    Plots a relative importance/coefficient chart. Tree ensembles (LightGBM, XGBoost,
    RandomForest) expose `feature_importances_`; linear models (LogisticRegression)
    expose `coef_`, whose absolute value is used as a proxy for importance here.
    SVM with an RBF kernel exposes neither (the decision boundary is not linear in
    the input space), so this plot is skipped for that case.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        xlabel = 'Relative Importance Score'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        xlabel = '|Coefficient| (standardized features)'
    else:
        print(f"[INFO] {model_name} exposes neither feature_importances_ nor coef_ "
              f"(e.g. RBF-kernel SVM) -- skipping the importance plot for this model.")
        return

    indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[indices]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features,
                palette='viridis', legend=False)
    plt.title(f'Feature Importances: {model_name} ({dataset_display_name}, illustrative single fit)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    out_path = out(f'{output_prefix}_feature_importance.png')
    plt.savefig(out_path, dpi=150)
    plt.close()  # Fix #3: close the figure instead of popping up a window to close by hand
    print(f"[OK] Saved '{out_path}'")


def plot_final_confusion_matrix(model, X_test, y_test, threshold, model_name, class_labels, dataset_display_name, output_prefix):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix: {model_name} ({dataset_display_name})\n(Illustrative single split, threshold={threshold:.3f})')
    plt.grid(False)
    out_path = out(f'{output_prefix}_confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.close()  # Fix #3: close the figure instead of popping up a window to close by hand
    print(f"[OK] Saved '{out_path}'")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"\n--- Clinical Metrics for {model_name} on {dataset_display_name} (illustrative single split) ---")
    print(f"Sensitivity / Recall (True Positive Rate): {sensitivity:.2%}")
    print(f"Specificity (True Negative Rate): {specificity:.2%}")
    print(f"Missed Patient Cases (False Negatives): {fn}")


def plot_f1_boxplot(results_df, dataset_display_name, output_prefix):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='augmentation', y='f1', hue='model')
    plt.title(f'F1-Score Distribution across {results_df["fold"].nunique()} Nested CV Outer Folds ({dataset_display_name})')
    plt.ylabel('F1-Score (outer test folds)')
    plt.xlabel('Augmentation Strategy')
    plt.tight_layout()
    out_path = out(f'{output_prefix}_nested_cv_f1_boxplot.png')
    plt.savefig(out_path, dpi=150)
    plt.close()  # Fix #3: close the figure instead of popping up a window to close by hand
    print(f"[OK] Saved '{out_path}'")


def plot_calibration_curve(model, X_test, y_test, model_name, dataset_display_name, output_prefix):
    """
    Fix #4 (v8): reliability diagram (predicted probability vs observed frequency) for the
    illustrative best-combo model, plus its Brier score -- a calibration claim ("this
    probability output can be trusted at face value") is a separate claim from F1/AUC and
    matters for a clinical-decision-support framing, since a clinician may read the
    predicted probability directly rather than just the thresholded yes/no call.
    """
    probs = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs)
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.plot(mean_pred, frac_pos, marker='o', label=model_name)
    plt.title(f'Calibration Curve ({dataset_display_name})\nBrier score = {brier:.3f} (lower is better)')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed frequency of positive class')
    plt.legend()
    plt.tight_layout()
    out_path = out(f'{output_prefix}_calibration_curve.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved '{out_path}' (Brier score={brier:.4f})")
    return brier


# --- 8. DATASET REGISTRY ---
# Each entry fully describes one dataset: how to load it, what its classes are called
# (for confusion-matrix labels), a human-readable name (for plot titles/console banners),
# and an output_prefix so every CSV/PNG this dataset produces gets its own unique filename
# and nothing from one dataset ever overwrites another's output.
DATASETS = {
    'pima': {
        'display_name': 'Pima Indians Diabetes',
        'loader': lambda: load_and_preprocess_pima('diabetes.csv'),
        'class_labels': ['Non-Diabetic', 'Diabetic'],
        'output_prefix': 'pima',
        # Fix #4 (v6): only Pima has columns where 0 is a physiologically impossible
        # missing-value sentinel -- see PIMA_ZERO_AS_MISSING_COLUMNS / apply_zero_as_missing_imputation.
        'zero_as_missing_columns': PIMA_ZERO_AS_MISSING_COLUMNS,
    },
    'heart': {
        'display_name': 'Heart Disease (UCI)',
        'loader': lambda: load_and_preprocess_heart('heart.csv'),
        'class_labels': ['No Disease', 'Disease'],
        'output_prefix': 'heart',
        # Deliberately empty: several columns here are categorical/binary flags where 0 is a
        # legitimate value (not missing data), so blanket zero-imputation isn't appropriate.
        'zero_as_missing_columns': [],
    },
    'breast_cancer': {
        'display_name': 'Breast Cancer Wisconsin',
        'loader': lambda: load_and_preprocess_breast_cancer('breast_cancer.csv'),
        'class_labels': ['Benign', 'Malignant'],
        'output_prefix': 'breast_cancer',
        # Deliberately empty: e.g. 'concavity_mean'/'concave points_mean' can genuinely be 0
        # for a smooth tumor -- 0 there is real signal, not missing data.
        'zero_as_missing_columns': [],
    },
}


# --- 9. PER-DATASET BENCHMARK PIPELINE ---
def run_benchmark_for_dataset(dataset_key, config):
    """
    Runs the full nested-CV benchmark + illustrative plots for a single dataset and
    saves every output file (CSV/PNG) under that dataset's own output_prefix, so results
    for pima / heart / breast_cancer never collide or overwrite one another.
    """
    display_name = config['display_name']
    class_labels = config['class_labels']
    prefix = config['output_prefix']

    dataset_start = time.time()
    print("\n" + "#" * 75)
    print(f"# DATASET: {display_name}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 75)

    set_seed(RANDOM_SEED)

    # 1. Load data
    X, y = config['loader']()
    feature_names = X.columns.tolist()

    # 2. Define classifiers as builder functions (fresh, unfitted instance each call)
    # NOTE: none of these set class_weight='balanced' / scale_pos_weight on purpose --
    # the 'none' augmentation condition is meant to be a true "no imbalance handling"
    # baseline. If a model auto-balanced internally, it would confound the comparison
    # against SMOTE / WGAN-GP (which handle imbalance via resampling instead).
    # Fix #6: each builder now takes a `seed` argument so every fold gets its own model
    # random_state (fold_seed) instead of every fold reusing the same constant RANDOM_SEED.
    # Fix #2: SVM's probability estimates now go through an explicit CalibratedClassifierCV
    # with cv=3 rather than SVC(probability=True)'s built-in (and non-configurable) 5-fold
    # Platt scaling -- same idea, but cheaper and its cost is visible/tunable here.
    model_builders = {
        'LightGBM': lambda seed: lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                                     random_state=seed, verbosity=-1),
        'XGBoost': lambda seed: XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4,
                                               random_state=seed, eval_metric='logloss'),
        'RF': lambda seed: RandomForestClassifier(n_estimators=200, max_depth=None,
                                                   random_state=seed, n_jobs=-1),
        'SVM': lambda seed: CalibratedClassifierCV(
            SVC(kernel='rbf', probability=False, random_state=seed), cv=3, method='sigmoid'),
        'LR': lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    }

    augmentations = ['none', 'smote', 'wgan_gp']

    # Fix #4: resolve the dataset's zero-as-missing column names (if any -- only Pima has
    # them, see DATASETS / PIMA_ZERO_AS_MISSING_COLUMNS) to positional indices, since X
    # becomes a plain ndarray inside run_nested_cv.
    zero_as_missing_columns = config.get('zero_as_missing_columns') or []
    if APPLY_ZERO_AS_MISSING_IMPUTATION and zero_as_missing_columns:
        impute_col_indices = [feature_names.index(c) for c in zero_as_missing_columns]
        print(f"[INFO] {display_name}: zero-as-missing median imputation ENABLED for columns: "
              f"{zero_as_missing_columns}")
    else:
        impute_col_indices = []

    # 3. Repeated k-fold nested CV (N_SPLITS splits x N_REPEATS repeats = total outer folds).
    #    The WGAN-GP config is now selected PER FOLD inside the loop (no pilot search).
    print("\n" + "=" * 75)
    print(f"[{display_name}] REPEATED {N_SPLITS}-FOLD NESTED CROSS-VALIDATION "
          f"({N_SPLITS} splits x {N_REPEATS} repeats = {N_SPLITS * N_REPEATS} outer folds)")
    print(f"[{display_name}] Inner {INNER_FOLDS}-fold CV inside every outer-train set: WGAN-GP config "
          f"({len(WGAN_GP_SEARCH_SPACE)} candidates), classifier hyperparameters and threshold, all from "
          f"pooled out-of-fold predictions")
    print("=" * 75)
    pred_sink = [] if SAVE_PREDICTIONS else None
    results_df, qc_per_fold = run_nested_cv(
        X, y, model_builders, augmentations,
        n_splits=N_SPLITS, n_repeats=N_REPEATS, inner_folds=INNER_FOLDS, base_seed=RANDOM_SEED,
        wgan_gp_search_space=WGAN_GP_SEARCH_SPACE, impute_zero_col_indices=impute_col_indices,
        feature_names=feature_names, search_n_iter=N_SEARCH_ITER, prediction_sink=pred_sink
    )
    if pred_sink is not None and pred_sink:
        pd.concat(pred_sink, ignore_index=True).to_csv(
            out(f'{prefix}_outer_test_predictions.csv.gz'), index=False, float_format='%.8g')
        print(f"[{display_name}] Saved outer-test predictions -> {prefix}_outer_test_predictions.csv.gz")

    # 3b. Per-fold WGAN-GP config selection frequencies + aggregated synthetic QC.
    label_to_config = {config_label(c): c for c in WGAN_GP_SEARCH_SPACE}
    sel = (results_df[results_df['augmentation'] == 'wgan_gp']
           .drop_duplicates('fold')['wgan_config'].value_counts())
    config_counts = {k: int(v) for k, v in sel.items()}
    sel.rename_axis('wgan_config').reset_index(name='n_folds_selected').to_csv(
        out(f'{prefix}_wgan_gp_config_selection.csv'), index=False)
    print(f"[{display_name}] WGAN-GP per-fold config selection counts: {config_counts}")
    wgan_gp_config = label_to_config[sel.index[0]] if len(sel) else None  # most frequent; plots only

    if qc_per_fold is not None:
        qc_per_fold.to_csv(out(f'{prefix}_wgan_gp_synthetic_qc_per_fold.csv'), index=False)
        wgan_gp_qc = (qc_per_fold.groupby('feature')
                      .agg(ks_stat=('ks_stat', 'mean'), ks_stat_std=('ks_stat', 'std'),
                           frac_synthetic_out_of_range=('frac_synthetic_out_of_range', 'mean'))
                      .reset_index().sort_values('ks_stat', ascending=False))
        wgan_gp_qc.to_csv(out(f'{prefix}_wgan_gp_synthetic_qc.csv'), index=False)
        n_flagged = int((wgan_gp_qc['frac_synthetic_out_of_range'] > 0.05).sum())
        n_ks = int((wgan_gp_qc['ks_stat'] > 0.30).sum())
        print(f"[WGAN-GP QC] Averaged over {qc_per_fold['fold'].nunique()} per-fold generators: "
              f"{n_flagged}/{len(wgan_gp_qc)} feature(s) with >5% synthetic values outside the real "
              f"min/max range; {n_ks}/{len(wgan_gp_qc)} with mean KS stat > 0.30.")
    else:
        wgan_gp_qc = None

    # 4. Summary table (mean +/- std across outer folds)
    summary = results_df.groupby(['augmentation', 'model']).agg(
        F1_mean=('f1', 'mean'), F1_std=('f1', 'std'),
        Recall_mean=('recall', 'mean'), Recall_std=('recall', 'std'),
        AUC_mean=('auc_roc', 'mean'), AUC_std=('auc_roc', 'std'),
        PR_AUC_mean=('pr_auc', 'mean'), PR_AUC_std=('pr_auc', 'std'),
        Brier_mean=('brier_score', 'mean'), Brier_std=('brier_score', 'std'),
        FN_mean=('false_negatives', 'mean'),
        F1_median=('f1', 'median'),
        Collapsed_folds=('collapsed', 'sum'),
    ).reset_index().sort_values('F1_mean', ascending=False)

    print("\n" + "=" * 75)
    print(f"BENCHMARK SUMMARY: {display_name.upper()} (mean +/- std over {N_SPLITS * N_REPEATS} outer folds)")
    print("=" * 75)
    print(summary.to_string(index=False))

    detailed_path = out(f'{prefix}_nested_cv_detailed_results.csv')
    summary_path = out(f'{prefix}_nested_cv_summary.csv')
    results_df.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved '{detailed_path}' and '{summary_path}'")

    # 5. Visual comparison across augmentation strategies
    plot_f1_boxplot(results_df, display_name, prefix)

    # 6. Illustrative single train/test fit for feature importance & confusion matrix plots.
    #    NOTE: this is for visualization only and is NOT how the benchmark numbers above
    #    were produced -- those come exclusively from the nested CV loop.
    best_row = summary.iloc[0]
    best_aug, best_model_name = best_row['augmentation'], best_row['model']
    print(f"\n[{display_name}] Generating illustrative plots for the best-performing combo: "
          f"{best_model_name} + {best_aug} (single held-out split, not used for scoring above)...")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_tr_arr, X_te_arr = X_tr.to_numpy(), X_te.to_numpy()
    if impute_col_indices:
        X_tr_arr, X_te_arr = apply_zero_as_missing_imputation(X_tr_arr, X_te_arr, impute_col_indices)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_arr)
    X_te_s = scaler.transform(X_te_arr)

    X_tr_aug, y_tr_aug = apply_augmentation(best_aug, X_tr_s, y_tr.to_numpy(), RANDOM_SEED,
                                             wgan_gp_config=wgan_gp_config)
    illustrative_model = build_tuned_model(
        best_model_name, model_builders[best_model_name],
        CLASSIC_MODEL_PARAM_GRIDS[best_model_name], X_tr_aug, y_tr_aug, RANDOM_SEED,
        n_iter=N_SEARCH_ITER, cv=SEARCH_CV_FOLDS
    )

    probs = illustrative_model.predict_proba(X_te_s)[:, 1]
    illustrative_thresh = find_best_threshold(y_te.to_numpy(), probs)  # only for plotting purposes

    plot_clinical_importance(illustrative_model, feature_names, best_model_name, display_name, prefix)
    plot_final_confusion_matrix(illustrative_model, X_te_s, y_te, illustrative_thresh, best_model_name,
                                 class_labels, display_name, prefix)
    illustrative_brier = plot_calibration_curve(illustrative_model, X_te_s, y_te, best_model_name,
                                                 display_name, prefix)

    dataset_elapsed_min = (time.time() - dataset_start) / 60
    print(f"\n[OK] {display_name}: evaluation successfully finalized. "
          f"Elapsed: {dataset_elapsed_min:.1f} min "
          f"(finished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # Fix #5 (v8): stash this dataset's facts for the end-of-run manifest.
    RUN_MANIFEST_INFO[dataset_key] = {
        'display_name': display_name,
        'n_rows': int(len(X)),
        'zero_as_missing_columns': zero_as_missing_columns if APPLY_ZERO_AS_MISSING_IMPUTATION else [],
        'wgan_gp_config_counts': config_counts,
        'wgan_gp_qc_flagged_features': (
            int((wgan_gp_qc['frac_synthetic_out_of_range'] > 0.05).sum()) if wgan_gp_qc is not None else None
        ),
        'wgan_gp_qc_total_features': len(feature_names) if wgan_gp_qc is not None else None,
        'best_model': best_model_name,
        'best_augmentation': best_aug,
        'best_f1_mean': float(best_row['F1_mean']),
        'best_auc_mean': float(best_row['AUC_mean']),
        'illustrative_brier_score': float(illustrative_brier),
        'elapsed_min': dataset_elapsed_min,
    }
    return summary


# --- 9b. RUN MANIFEST (Fix #5, v8): human-readable record of exactly how this run was configured ---
def _library_versions():
    """Best-effort library version info for the manifest; a missing package is just noted, not fatal."""
    import sklearn
    import imblearn
    versions = {
        'python': sys.version.split()[0],
        'torch': getattr(torch, '__version__', 'unknown'),
        'scikit-learn': sklearn.__version__,
        'imbalanced-learn': imblearn.__version__,
        'xgboost': __import__('xgboost').__version__,
        'lightgbm': lgb.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scipy': __import__('scipy').__version__,
    }
    return versions


def write_run_manifest(run_dir, cli_args, program_start, program_end, all_summaries):
    """
    Writes 'run_config.md' inside run_dir: every effective setting, library version, and
    per-dataset outcome for this run, in one human-readable file -- so a paper's Methods
    section can be checked against (or generated from) this instead of being retyped by
    hand from memory, and so re-running months later with different settings doesn't leave
    ambiguity about which numbers came from which configuration. Markdown, not JSON/YAML,
    per the agreed format.
    """
    lines = []
    lines.append(f"# Run manifest: {os.path.basename(__file__)}")
    lines.append("")
    lines.append(f"- **Started:** {datetime.fromtimestamp(program_start).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Finished:** {datetime.fromtimestamp(program_end).strftime('%Y-%m-%d %H:%M:%S')}")
    total_min = (program_end - program_start) / 60
    lines.append(f"- **Total elapsed:** {total_min:.1f} min "
                 f"({str(timedelta(minutes=total_min)).split('.')[0]})")
    lines.append(f"- **Device used for WGAN-GP:** {DEVICE}")
    lines.append("")

    lines.append("## CLI arguments used")
    lines.append("")
    for k, v in sorted(vars(cli_args).items()):
        lines.append(f"- `--{k.replace('_', '-')}`: `{v}`")
    lines.append("")

    lines.append("## Library versions")
    lines.append("")
    try:
        for name, ver in _library_versions().items():
            lines.append(f"- {name}: `{ver}`")
    except Exception as e:
        lines.append(f"(could not collect all library versions: {e})")
    lines.append("")

    lines.append("## Hyperparameter search spaces (classical models)")
    lines.append("")
    lines.append(f"- Randomized candidate search: `n_iter={N_SEARCH_ITER}` candidates per outer fold/model/augmentation, "
                 f"scored by pooled out-of-fold PR-AUC over an inner `{INNER_FOLDS}`-fold CV (augmentation fitted on each "
                 f"inner-train part only; scoring on real inner-val rows only)")
    for model_name, grid in CLASSIC_MODEL_PARAM_GRIDS.items():
        lines.append(f"  - **{model_name}**: `{grid}`")
    lines.append("")

    lines.append("## WGAN-GP search space (per outer fold; selected by pooled inner-CV out-of-fold PR-AUC)")
    lines.append("")
    for cfg in WGAN_GP_SEARCH_SPACE:
        lines.append(f"- `{cfg}`")
    lines.append("")

    lines.append("## Per-dataset results")
    lines.append("")
    for dataset_key, info in RUN_MANIFEST_INFO.items():
        lines.append(f"### {info['display_name']}")
        lines.append("")
        lines.append(f"- Rows: {info['n_rows']}")
        lines.append(f"- Zero-as-missing imputed columns: `{info['zero_as_missing_columns'] or 'none'}`")
        lines.append(f"- WGAN-GP per-fold config selection counts: `{info['wgan_gp_config_counts']}`")
        if info['wgan_gp_qc_total_features'] is not None:
            lines.append(f"- WGAN-GP synthetic QC: {info['wgan_gp_qc_flagged_features']}/"
                         f"{info['wgan_gp_qc_total_features']} feature(s) flagged "
                         f"(mean over per-fold generators: >5% synthetic values outside the real min/max range) "
                         f"-- see `{dataset_key}_wgan_gp_synthetic_qc.csv`")
        else:
            lines.append("- WGAN-GP synthetic QC: skipped (no synthetic samples generated)")
        lines.append(f"- Best combo: **{info['best_model']} + {info['best_augmentation']}** "
                     f"(F1={info['best_f1_mean']:.3f}, AUC={info['best_auc_mean']:.3f})")
        lines.append(f"- Illustrative-split Brier score: {info['illustrative_brier_score']:.4f}")
        lines.append(f"- Elapsed: {info['elapsed_min']:.1f} min")
        lines.append("")

    lines.append("## Documented methodological limitations")
    lines.append("")
    lines.append(f"- Inner CV: {INNER_FOLDS}-fold stratified CV inside every outer-train set; its pooled "
                 "out-of-fold predictions drive WGAN-GP config selection, classifier hyperparameter "
                 "selection and the F1-optimal decision threshold. The final model is refit on the whole "
                 "(augmented) outer-train set; its WGAN-GP is a different random draw than the inner ones, "
                 "so a small probability-scale mismatch between threshold and final model can remain "
                 "(see the `collapsed`, `oof_pos_rate` and `test_pos_rate` columns).")
    lines.append("- WGAN-GP's latent_dim is fixed at 16 and not searched; the 4 candidate configs are "
                 "scored with a fixed default LightGBM, not with each final classifier.")
    lines.append("- The classifier hyperparameter search samples n_iter candidates from small grids "
                 "(not exhaustive) and scores PR-AUC, not the F1 at the final threshold.")
    lines.append("- The 45 outer folds come from repeated stratified k-fold CV over the same "
                 "dataset and are correlated, not independent -- relevant if p-values from "
                 "the companion statistical-analysis script are reported (see its own "
                 "REPEATED_CV_CAVEAT).")
    lines.append("")

    manifest_path = os.path.join(run_dir, 'run_config.md')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved run manifest to '{manifest_path}'")
    return manifest_path


# --- 10. EXECUTION PIPELINE (RUNS ALL DATASETS) ---
def parse_args():
    """Fix #8 (v8): CLI overrides for the settings you'd otherwise have to edit in the code
    to reuse this pipeline on a new run/study. Omit everything and behavior matches v7
    (all 3 datasets, n_splits=3, n_repeats=15, seed=42, imputation on)."""
    parser = argparse.ArgumentParser(
        description="Repeated nested-CV benchmark: augmentation strategy x model, "
                    "across Pima / Heart / Breast Cancer."
    )
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()), default=None,
                        help="Which datasets to run (default: all of them).")
    parser.add_argument('--n-splits', type=int, default=N_SPLITS,
                        help="Outer CV splits per repeat.")
    parser.add_argument('--n-repeats', type=int, default=N_REPEATS,
                        help="How many times to repeat the outer split.")
    parser.add_argument('--search-iter', type=int, default=N_SEARCH_ITER,
                        help="RandomizedSearchCV candidates per fold/model/augmentation.")
    parser.add_argument('--inner-folds', type=int, default=INNER_FOLDS,
                        help="Inner CV folds inside every outer-train set (config/hyperparameter "
                             "search and threshold selection). GAN trainings per outer fold = 4 x K + 1.")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Base random seed.")
    parser.add_argument('--save-predictions', action='store_true',
                        help="Also save every outer-test predicted probability (needed for the pooled "
                             "calibration analysis in calibration_analysis_v1.py).")
    parser.add_argument('--deterministic', action='store_true',
                        help="Ask PyTorch/cuBLAS/cuDNN for deterministic GPU kernels (slower). Use it to "
                            "check that WGAN-GP results are repeatable run-to-run for a fixed --seed.")
    parser.add_argument('--no-impute', action='store_true',
                        help="Disable zero-as-missing median imputation for Pima.")
    parser.add_argument('--output-root', type=str, default=None,
                        help="Parent folder for timestamped run folders "
                            "(default: '<script_dir>/runs').")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # Apply CLI overrides to the module-level globals every downstream function reads.
    N_SPLITS = cli_args.n_splits
    N_REPEATS = cli_args.n_repeats
    N_SEARCH_ITER = cli_args.search_iter
    INNER_FOLDS = cli_args.inner_folds
    SAVE_PREDICTIONS = cli_args.save_predictions
    RANDOM_SEED = cli_args.seed
    if cli_args.deterministic:
        # cuBLAS needs this env var before its first use; warn_only keeps any op without a
        # deterministic implementation from crashing the run (it logs a warning instead).
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if cli_args.no_impute:
        APPLY_ZERO_AS_MISSING_IMPUTATION = False

    selected_datasets = {k: v for k, v in DATASETS.items()
                         if cli_args.datasets is None or k in cli_args.datasets}

    # Fix #7 (v8): fresh timestamped folder for this run's outputs -- never overwrites a
    # previous run. Dataset CSV caches (diabetes.csv etc.) stay in the stable DATA_DIR.
    output_root = cli_args.output_root or os.path.join(DATA_DIR, 'runs')
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(output_root, f'run_{run_timestamp}')
    os.makedirs(RUN_DIR, exist_ok=True)

    # Fix #10 (v8): mirror all console output into run.log inside RUN_DIR from this point on.
    log_path = setup_run_logging(RUN_DIR)

    program_start = time.time()
    print(f"[INFO] Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] This run's outputs (CSV/PNG/log/manifest) will be saved to: {RUN_DIR}")
    print(f"[INFO] Dataset CSV cache (shared across runs): {DATA_DIR}")
    print(f"[INFO] Console output is also being written to: {log_path}")
    print(f"[INFO] Effective config: n_splits={N_SPLITS} n_repeats={N_REPEATS} "
          f"search_iter={N_SEARCH_ITER} inner_folds={INNER_FOLDS} seed={RANDOM_SEED} "
          f"impute={APPLY_ZERO_AS_MISSING_IMPUTATION} datasets={list(selected_datasets.keys())}")

    all_summaries = {}
    for dataset_key, config in selected_datasets.items():
        all_summaries[dataset_key] = run_benchmark_for_dataset(dataset_key, config)

    print("\n" + "#" * 75)
    print("# ALL DATASETS COMPLETE -- BEST COMBO PER DATASET")
    print("#" * 75)
    for dataset_key, summary in all_summaries.items():
        best_row = summary.iloc[0]
        display_name = selected_datasets[dataset_key]['display_name']
        print(f"{display_name:30s} -> {best_row['model']:18s} + {best_row['augmentation']:8s} "
              f"| F1={best_row['F1_mean']:.3f} | AUC={best_row['AUC_mean']:.3f}")

    program_end = time.time()
    total_elapsed_min = (program_end - program_start) / 60
    print(f"\n[OK] Evaluation successfully finalized for all datasets. Data points are ready.")
    print(f"[INFO] Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"| Total elapsed: {total_elapsed_min:.1f} min "
          f"({str(timedelta(minutes=total_elapsed_min)).split('.')[0]})")

    # Fix #5 (v8): write the run manifest last, once every dataset's info has been collected.
    write_run_manifest(RUN_DIR, cli_args, program_start, program_end, all_summaries)
