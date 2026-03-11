"""
Train and evaluate ML models for exoplanet habitability prediction.

Models: Random Forest Regressor, Gradient Boosting Regressor.
Explainability: SHAP TreeExplainer.
Validation: Solar System bodies + PHL expert-curated habitable candidates.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ASSET_DIR = os.path.join(BASE_DIR, 'assets')

FEATURE_COLS = [
    'pl_bmasse',              # planet mass
    'pl_rade',                # planet radius
    'pl_orbper',              # orbital period
    'pl_orbsmax',             # semi-major axis
    'pl_orbeccen',            # eccentricity
    'pl_eqt',                 # equilibrium temperature
    'pl_dens',                # density
    'st_teff',                # stellar temperature
    'st_lum',                 # stellar luminosity
    'st_mass',                # stellar mass
    'st_rad',                 # stellar radius
    'escape_velocity',        # escape velocity
    'stellar_flux',           # stellar flux at planet
    'in_hz',                  # in habitable zone flag (eccentricity-aware)
    'is_rocky',               # rocky planet flag
    'esi',                    # earth similarity index
    'orbit_stability',        # orbital eccentricity penalty (Bolmont+ 2016)
    'spectral_suitability',   # stellar spectrum for photosynthesis
    'stellar_suitability',    # stellar lifetime + flare risk
]

TARGET_COL = 'habitability_score'


def load_training_data():
    """Load the final dataset and prepare features/target."""
    path = os.path.join(DATA_DIR, 'exoplanets_final.csv')
    df = pd.read_csv(path)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    names = df['pl_name'].copy()

    print(f"Full dataset: {len(df)} planets")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Missing values per feature:")
    for col in FEATURE_COLS:
        n_miss = X[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} ({100*n_miss/len(X):.1f}%)")

    return X, y, names, df


def train_models(X, y):
    """Train Random Forest and Gradient Boosting, return best model."""
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
    }
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    rf.fit(X_train, y_train)
    rf_best = rf.best_estimator_
    rf_pred = rf_best.predict(X_test)

    rf_metrics = {
        'model': 'RandomForest',
        'best_params': rf.best_params_,
        'r2': r2_score(y_test, rf_pred),
        'mse': mean_squared_error(y_test, rf_pred),
        'mae': mean_absolute_error(y_test, rf_pred),
        'cv_r2_mean': rf.best_score_,
    }
    print(f"  Best params: {rf.best_params_}")
    print(f"  R²: {rf_metrics['r2']:.4f}")
    print(f"  MSE: {rf_metrics['mse']:.6f}")
    print(f"  MAE: {rf_metrics['mae']:.4f}")
    print(f"  CV R² (mean): {rf_metrics['cv_r2_mean']:.4f}")

    # --- Gradient Boosting ---
    print("\n--- Gradient Boosting ---")
    gb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    gb.fit(X_train, y_train)
    gb_best = gb.best_estimator_
    gb_pred = gb_best.predict(X_test)

    gb_metrics = {
        'model': 'GradientBoosting',
        'best_params': {k: v if not isinstance(v, np.integer) else int(v) for k, v in gb.best_params_.items()},
        'r2': r2_score(y_test, gb_pred),
        'mse': mean_squared_error(y_test, gb_pred),
        'mae': mean_absolute_error(y_test, gb_pred),
        'cv_r2_mean': gb.best_score_,
    }
    print(f"  Best params: {gb.best_params_}")
    print(f"  R²: {gb_metrics['r2']:.4f}")
    print(f"  MSE: {gb_metrics['mse']:.6f}")
    print(f"  MAE: {gb_metrics['mae']:.4f}")
    print(f"  CV R² (mean): {gb_metrics['cv_r2_mean']:.4f}")

    # Pick best model
    if gb_metrics['r2'] > rf_metrics['r2']:
        best_model = gb_best
        best_name = 'GradientBoosting'
        best_metrics = gb_metrics
    else:
        best_model = rf_best
        best_name = 'RandomForest'
        best_metrics = rf_metrics

    print(f"\n>>> Best model: {best_name} (R² = {best_metrics['r2']:.4f})")

    return best_model, imputer, X_test, y_test, [rf_metrics, gb_metrics], best_name


def compute_shap(model, X, imputer, n_background=200):
    """Compute SHAP values for explainability."""
    print("\nComputing SHAP values...")
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_imp, show=False, max_display=16)
    plt.tight_layout()
    path = os.path.join(ASSET_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved SHAP summary plot to {path}")

    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_imp, plot_type='bar', show=False, max_display=16)
    plt.tight_layout()
    path = os.path.join(ASSET_DIR, 'shap_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved SHAP importance plot to {path}")

    # Mean absolute SHAP per feature
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    print("\n  Feature importance (mean |SHAP|):")
    for feat, val in mean_shap.items():
        print(f"    {feat:22s}: {val:.5f}")

    return shap_values, mean_shap


def validate_known_planets(model, imputer):
    """Validate model predictions against known Solar System and exoplanet benchmarks."""
    print("\n--- Validation on known planets ---")

    # Solar System + key exoplanets with known parameters
    # Columns: name + FEATURE_COLS (19 features)
    # orbit_stability, spectral_suitability, stellar_suitability computed
    # from physical formulas in features.py
    known = pd.DataFrame([
        #              mass  rad   per     sma    ecc   eqt  dens  st_t   st_l   st_m  st_r  esc   flux  hz rky esi  orb_s spec  stel
        ["Earth",      1.0,  1.0,  365.25, 1.0,   0.017,255, 5.51, 5778, 0.0,   1.0,  1.0,  11.2, 1.0,  1, 1, 1.0, 1.0,  0.87, 1.0 ],
        ["Mars",       0.107,0.532,687.0,  1.524, 0.093,210, 3.93, 5778, 0.0,   1.0,  1.0,  5.03, 0.43, 1, 1, 0.73,0.95, 0.87, 1.0 ],
        ["Venus",      0.815,0.949,224.7,  0.723, 0.007,232, 5.24, 5778, 0.0,   1.0,  1.0,  10.36,1.91, 0, 1, 0.78,0.0,  0.87, 1.0 ],
        ["Jupiter",    317.8,11.21,4333.0, 5.2,   0.049,110, 1.33, 5778, 0.0,   1.0,  1.0,  59.5, 0.037,0, 0, 0.29,0.0,  0.87, 1.0 ],
        ["TRAPPIST-1 e",0.692,0.920,6.1,   0.029, 0.005,230, 5.65, 2566, -2.57, 0.089,0.121,10.2, 0.65, 1, 1, 0.85,1.0,  0.27, 0.07],
        ["K2-18 b",    8.92, 2.37, 32.94,  0.143, 0.0,  284, 3.68, 3457, -1.51, 0.359,0.411,8.44, 1.26, 0, 1, 0.77,0.0,  0.59, 0.52],
        ["Kepler-442 b",2.34,1.34, 112.3,  0.409, 0.04, 233, 5.57, 4402, -0.78, 0.609,0.598,9.36, 0.70, 1, 1, 0.84,0.99, 0.92, 0.93],
        ["Proxima Cen b",1.07,1.08, 11.19, 0.049, 0.11, 234, 4.95, 3042, -2.54, 0.122,0.154,10.8, 0.68, 1, 1, 0.86,0.93, 0.43, 0.09],
    ], columns=['name'] + FEATURE_COLS)

    X_known = known[FEATURE_COLS].astype(float)
    X_known_imp = pd.DataFrame(imputer.transform(X_known), columns=FEATURE_COLS)
    preds = model.predict(X_known_imp)

    results = pd.DataFrame({
        'Planet': known['name'],
        'Predicted Score': preds,
    })
    results['Expected'] = ['High (life exists)', 'Medium (past water)',
                           'Low (runaway greenhouse)', 'Very low (gas giant)',
                           'High (top candidate)', 'Medium-High (JWST target)',
                           'High (HZ super-Earth)', 'High (nearest HZ)']

    print(results.to_string(index=False))

    # Save validation results
    os.makedirs(ASSET_DIR, exist_ok=True)
    results.to_csv(os.path.join(ASSET_DIR, 'validation_results.csv'), index=False)

    return results


def validate_phl_labels(model, imputer, df):
    """Check model scores against PHL expert habitability labels."""
    print("\n--- PHL Expert Label Validation ---")
    has_label = df[df['phl_habitable'].notna()].copy()
    X_lab = has_label[FEATURE_COLS]
    X_lab_imp = pd.DataFrame(imputer.transform(X_lab), columns=FEATURE_COLS, index=X_lab.index)
    preds = model.predict(X_lab_imp)
    has_label = has_label.copy()
    has_label['predicted_score'] = preds

    for label, desc in [(0, "Not habitable"), (1, "Conservative"), (2, "Optimistic")]:
        subset = has_label[has_label['phl_habitable'] == label]['predicted_score']
        if len(subset) > 0:
            print(f"  {desc} (n={len(subset)}): mean={subset.mean():.4f}, "
                  f"median={subset.median():.4f}, min={subset.min():.4f}, max={subset.max():.4f}")

    # Separation quality
    hab_scores = has_label[has_label['phl_habitable'] > 0]['predicted_score']
    non_scores = has_label[has_label['phl_habitable'] == 0]['predicted_score']
    if len(hab_scores) > 0 and len(non_scores) > 0:
        sep = hab_scores.mean() - non_scores.mean()
        print(f"\n  Mean score separation (habitable - non): {sep:.4f}")
        print(f"  Habitable planets scoring > 0.3: {(hab_scores > 0.3).sum()}/{len(hab_scores)}")


def save_model(model, imputer, metrics, model_name):
    """Save model, imputer, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))

    # Save metadata
    meta = {
        'model_name': model_name,
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'metrics': [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                      for k, v in m.items()} for m in metrics],
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved model to {MODEL_DIR}/")
    print(f"  best_model.pkl, imputer.pkl, metadata.json")


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)

    X, y, names, df = load_training_data()
    best_model, imputer, X_test, y_test, all_metrics, best_name = train_models(X, y)
    shap_values, mean_shap = compute_shap(best_model, X, imputer)
    val_results = validate_known_planets(best_model, imputer)
    validate_phl_labels(best_model, imputer, df)
    save_model(best_model, imputer, all_metrics, best_name)

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
