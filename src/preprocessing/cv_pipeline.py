"""
src/preprocessing/cv_pipeline.py

Shared fold-safe preprocessing for cross-validation across all stages.
Fits MICE, scaling, and label encoding fresh inside every fold to avoid
leakage between train/validation splits within CV.

Originally authored in 04_cross_validation.ipynb; moved here so Stage 2+
notebooks can reuse it without copy-pasting.

Binary markers (BINARY_COLS) are clipped to [0,1] after MICE but NOT
rounded to a hard 0/1. Rounding was tested and found to destroy real
information -- for markers with a genuinely ambiguous ~50% imputed
probability, rounding manufactured fabricated associations with no real
clinical signal (e.g. ANA in PsA: true imputed rate ~46.8%, rounded to
18.6%). Keeping the continuous [0,1] estimate improved Random Forest
Macro-F1 by +0.0064, with the largest gains on AS, Normal, and Sjogren's
recall. See 02_preprocessing.ipynb for the investigation.
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

BINARY_COLS = ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']
MICE_COLS = ['ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27', 'ANA',
             'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']
CONTINUOUS_COLS = ['ESR', 'CRP', 'RF', 'Anti-CCP', 'C3', 'C4']

RAW_FEATURES = ['Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27',
                'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']
EXT_FEATURES = RAW_FEATURES + ['RF_was_missing', 'Anti-CCP_was_missing',
                                'Inflammation_Score', 'C3_C4_Ratio', 'Autoantibody_Count']

# NOTE: this was derived from SHAP on the OLD (pre-fix) preprocessing --
# needs to be re-verified against the corrected data before Stage 3 uses it.
# See 05_stage2_shap_reduction.ipynb.
REDUCED_FEATURES = ['ESR', 'RF', 'Anti-CCP', 'CRP', 'HLA-B27', 'C3', 'C4',
                     'Inflammation_Score', 'ANA', 'Anti-Ro', 'Anti-La']


def preprocess_fold(fold_train_df: pd.DataFrame, fold_val_df: pd.DataFrame):
    """
    Fits every learned preprocessing step (MICE, scaler, label encoder) on
    fold_train ONLY, then applies transform-only to fold_val. Nothing here
    ever calls .fit() on fold_val -- that's the entire point of doing this
    per-fold instead of reusing globally-fit versions.

    Returns X_raw_tr, X_raw_va, X_ext_tr, X_ext_va, y_tr, y_va.
    For the reduced feature set, slice the returned X_ext_* frames down to
    cv_pipeline.REDUCED_FEATURES afterward -- no separate scaling needed
    since those columns are already scaled correctly within X_ext.
    """
    ft, fv = fold_train_df.copy(), fold_val_df.copy()

    # MICE -- fit on fold train, transform both.
    # max_iter=50: max_iter=15 was tested and does not converge (true
    # convergence is ~iteration 17); 15 produced unstable, non-reproducible
    # downstream results.
    imputer = IterativeImputer(random_state=42, max_iter=50)
    ft[MICE_COLS] = imputer.fit_transform(ft[MICE_COLS])
    fv[MICE_COLS] = imputer.transform(fv[MICE_COLS])

    # Binary markers: clip to valid [0,1] range but do NOT round to hard
    # 0/1 -- see module docstring for why.
    for col in BINARY_COLS:
        ft[col] = ft[col].clip(0, 1)
        fv[col] = fv[col].clip(0, 1)
    for col in CONTINUOUS_COLS:
        ft[col] = ft[col].clip(lower=0)
        fv[col] = fv[col].clip(lower=0)

    for d in (ft, fv):
        # Inflammation_Score = ESR + CRP: an ad-hoc composite, NOT a
        # validated clinical score -- refer to it as such in the paper.
        d['Inflammation_Score'] = d['ESR'] + d['CRP']

        # C3_C4_Ratio: exploratory, hypothesis-driven (complement
        # consumption differs in SLE). Guard against div-by-zero.
        d['C3_C4_Ratio'] = d['C3'] / d['C4'].replace(0, pd.NA)

        # Autoantibody_Count: sum of the 6 binary markers' continuous
        # [0,1] estimates (not a count of hard positives, since binary
        # markers are no longer rounded). Exploratory aggregate feature,
        # not a validated diagnostic index.
        d['Autoantibody_Count'] = d[BINARY_COLS].sum(axis=1)

    le = LabelEncoder()
    ft['Disease_encoded'] = le.fit_transform(ft['Disease'])
    fv['Disease_encoded'] = le.transform(fv['Disease'])

    X_raw_tr, X_raw_va = ft[RAW_FEATURES].copy(), fv[RAW_FEATURES].copy()
    X_ext_tr, X_ext_va = ft[EXT_FEATURES].copy(), fv[EXT_FEATURES].copy()
    y_tr, y_va = ft['Disease_encoded'], fv['Disease_encoded']

    scaler_raw = StandardScaler()
    X_raw_tr[CONTINUOUS_COLS + ['Age']] = scaler_raw.fit_transform(X_raw_tr[CONTINUOUS_COLS + ['Age']])
    X_raw_va[CONTINUOUS_COLS + ['Age']] = scaler_raw.transform(X_raw_va[CONTINUOUS_COLS + ['Age']])

    ext_continuous = CONTINUOUS_COLS + ['Age', 'Inflammation_Score', 'C3_C4_Ratio', 'Autoantibody_Count']
    scaler_ext = StandardScaler()
    X_ext_tr[ext_continuous] = scaler_ext.fit_transform(X_ext_tr[ext_continuous])
    X_ext_va[ext_continuous] = scaler_ext.transform(X_ext_va[ext_continuous])

    return X_raw_tr, X_raw_va, X_ext_tr, X_ext_va, y_tr, y_va