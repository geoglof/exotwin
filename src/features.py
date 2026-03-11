"""
Feature engineering for exoplanet habitability prediction.

Computes derived physical properties and a composite habitability score
based on established astrophysical models:
- Habitable Zone: Kopparapu et al. (2013), eccentricity-aware per Bolmont et al. (2016)
- Earth Similarity Index: Schulze-Makuch et al. (2011)
- Spectral suitability for photosynthesis: Lingam & Loeb (2018), Meadows & Barnes (2018)
- Stellar suitability (lifetime + flare activity): Meadows & Barnes (2018)
"""

import numpy as np
import pandas as pd
import os

EARTH = {
    'mass': 1.0,        # Earth masses
    'radius': 1.0,      # Earth radii
    'density': 5.51,    # g/cm^3
    'eqt': 255.0,       # K (equilibrium temperature)
    'escape_vel': 11.2, # km/s
}

R_EARTH_KM = 6371.0
M_EARTH_KG = 5.972e24
G_CONST = 6.674e-11


def compute_density(df):
    """Compute bulk density from mass and radius where missing.
    density = mass / volume, with mass in Earth masses and radius in Earth radii.
    Result in g/cm^3.
    """
    mask = df['pl_dens'].isna() & df['pl_bmasse'].notna() & df['pl_rade'].notna()
    volume_ratio = df.loc[mask, 'pl_rade'] ** 3
    df.loc[mask, 'pl_dens'] = EARTH['density'] * df.loc[mask, 'pl_bmasse'] / volume_ratio
    return df


def compute_escape_velocity(df):
    """Escape velocity in km/s from mass (Earth masses) and radius (Earth radii).
    v_esc = sqrt(2GM/R)
    """
    mask = df['pl_bmasse'].notna() & df['pl_rade'].notna()
    mass_ratio = df.loc[mask, 'pl_bmasse']
    radius_ratio = df.loc[mask, 'pl_rade']
    df.loc[mask, 'escape_velocity'] = EARTH['escape_vel'] * np.sqrt(mass_ratio / radius_ratio)
    return df


def compute_stellar_luminosity(df):
    """Estimate luminosity from stellar mass where missing.
    Rough main-sequence approximation: L ~ M^3.5 (for 0.2 < M < 2 solar masses).
    """
    mask = df['st_lum'].isna() & df['st_mass'].notna()
    df.loc[mask, 'st_lum'] = np.log10(df.loc[mask, 'st_mass'] ** 3.5)
    return df


def compute_stellar_flux(df):
    """Stellar flux at planet's orbit (relative to Earth's solar flux).
    F = L / a^2 where L is in solar luminosities and a in AU.
    """
    mask = df['st_lum'].notna() & df['pl_orbsmax'].notna()
    lum_linear = 10 ** df.loc[mask, 'st_lum']
    df.loc[mask, 'stellar_flux'] = lum_linear / (df.loc[mask, 'pl_orbsmax'] ** 2)
    return df


def compute_habitable_zone(df):
    """Habitable zone boundaries using Kopparapu et al. (2013) coefficients.
    Eccentricity-aware: checks perihelion/aphelion against HZ boundaries
    following Bolmont et al. (2016) on limits of the mean flux approximation.
    """
    mask = df['st_teff'].notna() & df['st_lum'].notna()
    teff = df.loc[mask, 'st_teff']
    lum = 10 ** df.loc[mask, 'st_lum']

    t_star = teff - 5780.0

    s_eff_inner = 1.0146 + 8.1884e-5 * t_star + 1.9394e-9 * t_star**2
    s_eff_outer = 0.3507 + 5.9578e-5 * t_star + 1.6707e-9 * t_star**2

    df.loc[mask, 'hz_inner'] = np.sqrt(lum / s_eff_inner)
    df.loc[mask, 'hz_outer'] = np.sqrt(lum / s_eff_outer)

    hz_mask = mask & df['pl_orbsmax'].notna()

    ecc = df.loc[hz_mask, 'pl_orbeccen'].fillna(0).clip(lower=0)
    a = df.loc[hz_mask, 'pl_orbsmax']
    inner = df.loc[hz_mask, 'hz_inner']
    outer = df.loc[hz_mask, 'hz_outer']

    sma_in_hz = (a >= inner) & (a <= outer)
    df.loc[hz_mask, 'in_hz'] = sma_in_hz.astype(float)

    # Orbit stability: continuous (0-1) score combining HZ membership with
    # eccentricity penalty. Highly eccentric orbits cause flux variations
    # that overwhelm planetary thermal inertia (Bolmont et al. 2016).
    # Perihelion/aphelion excursions beyond HZ are captured here as a
    # soft penalty rather than in the binary in_hz flag.
    perihelion = a * (1 - ecc)
    aphelion = a * (1 + ecc)
    hz_tolerance = 0.05 * (outer - inner)
    full_in_hz = (perihelion >= inner - hz_tolerance) & (aphelion <= outer + hz_tolerance)
    ecc_penalty = np.exp(-0.5 * (ecc / 0.3) ** 2)

    orbit_stab = np.where(
        full_in_hz, ecc_penalty,
        np.where(sma_in_hz, 0.5 * ecc_penalty, 0.0)
    )
    df.loc[hz_mask, 'orbit_stability'] = orbit_stab

    return df


def compute_esi(df):
    """Earth Similarity Index (Schulze-Makuch et al. 2011).
    ESI = product of (1 - |x - x_earth| / (x + x_earth))^(w/n) for each parameter.
    Uses radius, density, escape velocity, and equilibrium temperature.
    """
    params = [
        ('pl_rade',          EARTH['radius'],     0.57),
        ('pl_dens',          EARTH['density'],     1.07),
        ('escape_velocity',  EARTH['escape_vel'],  0.70),
        ('pl_eqt',           EARTH['eqt'],         5.58),
    ]
    n = len(params)
    esi = pd.Series(1.0, index=df.index)
    valid = pd.Series(True, index=df.index)

    for col, earth_val, weight in params:
        if col not in df.columns:
            valid[:] = False
            continue
        mask = df[col].notna()
        valid &= mask
        ratio = ((df[col] - earth_val) / (df[col] + earth_val)).abs()
        component = (1 - ratio) ** (weight / n)
        esi *= component.fillna(1.0)

    df['esi'] = np.where(valid, esi, np.nan)
    return df


def compute_is_rocky(df):
    """Flag planets likely to be rocky (density > 3.5 g/cm^3 or radius < 1.6 R_earth)."""
    rocky_by_dens = df['pl_dens'].notna() & (df['pl_dens'] > 3.5)
    rocky_by_rad = df['pl_rade'].notna() & (df['pl_rade'] < 1.6)
    df['is_rocky'] = (rocky_by_dens | rocky_by_rad).astype(float)
    df.loc[df['pl_dens'].isna() & df['pl_rade'].isna(), 'is_rocky'] = np.nan
    return df


def compute_equilibrium_temp(df):
    """Estimate equilibrium temperature where missing.
    T_eq = T_star * sqrt(R_star / (2 * a)) * (1 - albedo)^0.25
    Assumes albedo = 0.3 (Earth-like).
    """
    mask = (df['pl_eqt'].isna() &
            df['st_teff'].notna() &
            df['st_rad'].notna() &
            df['pl_orbsmax'].notna())

    albedo = 0.3
    r_star_au = df.loc[mask, 'st_rad'] * 0.00465047  # Solar radii to AU
    df.loc[mask, 'pl_eqt'] = (
        df.loc[mask, 'st_teff'] *
        np.sqrt(r_star_au / (2 * df.loc[mask, 'pl_orbsmax'])) *
        (1 - albedo) ** 0.25
    )
    return df


def compute_spectral_suitability(df):
    """Spectral suitability for photosynthesis based on stellar temperature.

    G/K-type stars (4000-6000 K) provide optimal photosynthetically active
    radiation. Cool M-dwarfs peak in near-IR, limiting photosynthetic
    productivity (Lingam & Loeb 2018, ApJ 846). Hot F/A-type stars emit
    dangerous UV that can sterilize surfaces (Meadows & Barnes 2018).
    """
    mask = df['st_teff'].notna()
    teff = df.loc[mask, 'st_teff']

    score = np.exp(-0.5 * ((teff - 5000) / 1500) ** 2)
    uv_penalty = np.where(teff > 7500, np.exp(-(teff - 7500) / 1000), 1.0)

    df.loc[mask, 'spectral_suitability'] = score * uv_penalty
    return df


def compute_stellar_suitability(df):
    """Stellar suitability combining main-sequence lifetime and flare risk.

    Low-mass M-dwarfs (< 0.5 M_sun) have intense flare activity that strips
    atmospheres and irradiates surfaces. High-mass stars (> 2 M_sun) exhaust
    hydrogen too fast for life to emerge (lifetime < 1 Gyr).
    Optimal range: ~0.5-1.2 M_sun (late K to early G).
    Refs: Meadows & Barnes (2018), Kopparapu et al. (2013).
    """
    mask = df['st_mass'].notna()
    mass = df.loc[mask, 'st_mass']

    lifetime_gyr = 10.0 * mass ** (-2.5)
    lifetime_score = np.clip(lifetime_gyr / 4.0, 0, 1)

    flare_score = 1 / (1 + np.exp(-10 * (mass - 0.35)))

    df.loc[mask, 'stellar_suitability'] = lifetime_score * flare_score
    return df


def compute_habitability_score(df):
    """Composite habitability score (0-1), weighted combination of factors.

    Components (weights sum to 1.0):
    - in_hz (0 or 1):                HZ membership (eccentricity-aware)     0.22
    - esi (0-1):                     Earth Similarity Index                 0.20
    - temp_score (0-1):              Surface temperature suitability        0.15
    - atmo_score (0-1):              Atmosphere retention capability        0.12
    - orbit_stability (0-1):         Orbital eccentricity penalty           0.08
    - spectral_suitability (0-1):    Stellar spectrum for photosynthesis    0.08
    - stellar_suitability (0-1):     Stellar lifetime + flare risk          0.07
    - rocky_score (0 or 1):          Rocky composition                      0.08
    """
    hz = df['in_hz'].fillna(0)
    esi = df['esi'].fillna(0)

    temp = df['pl_eqt'].fillna(0)
    temp_score = np.exp(-0.5 * ((temp - 265) / 60) ** 2)
    temp_score = np.where(df['pl_eqt'].isna(), 0, temp_score)

    esc = df['escape_velocity'].fillna(0)
    atmo_score = 1 / (1 + np.exp(-2 * (esc - 5)))
    atmo_score = np.where(df['escape_velocity'].isna(), 0, atmo_score)

    rocky = df['is_rocky'].fillna(0)
    orbit_stab = df['orbit_stability'].fillna(0)
    spectral = df['spectral_suitability'].fillna(0)
    stellar = df['stellar_suitability'].fillna(0)

    raw = (0.22 * hz
           + 0.20 * esi
           + 0.15 * temp_score
           + 0.12 * atmo_score
           + 0.08 * orbit_stab
           + 0.08 * spectral
           + 0.07 * stellar
           + 0.08 * rocky)

    has_any_data = (
        df['in_hz'].notna() | df['esi'].notna() |
        df['pl_eqt'].notna() | df['escape_velocity'].notna() |
        df['is_rocky'].notna()
    )
    score = np.where(has_any_data, raw, np.nan)

    df['habitability_score'] = score
    return df


def run_feature_engineering(input_path=None, output_path=None):
    """Run the full feature engineering pipeline."""
    if input_path is None:
        input_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'exoplanets_clean.csv'
        )
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'exoplanets_features.csv'
        )

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} planets from {input_path}")

    steps = [
        ("Computing density",                compute_density),
        ("Computing stellar luminosity",     compute_stellar_luminosity),
        ("Computing equilibrium temp",       compute_equilibrium_temp),
        ("Computing escape velocity",        compute_escape_velocity),
        ("Computing stellar flux",           compute_stellar_flux),
        ("Computing habitable zone",         compute_habitable_zone),
        ("Computing rocky flag",             compute_is_rocky),
        ("Computing ESI",                    compute_esi),
        ("Computing spectral suitability",   compute_spectral_suitability),
        ("Computing stellar suitability",    compute_stellar_suitability),
        ("Computing habitability score",     compute_habitability_score),
    ]

    for desc, func in steps:
        print(f"  {desc}...")
        df = func(df)

    df.to_csv(output_path, index=False)
    print(f"\nSaved feature-engineered data to {output_path}")
    print(f"  Total planets: {len(df)}")
    print(f"  With habitability score: {df['habitability_score'].notna().sum()}")
    print(f"  In habitable zone: {int(df['in_hz'].sum()) if 'in_hz' in df.columns else 'N/A'}")
    print(f"  With ESI: {df['esi'].notna().sum()}")

    print(f"\nTop 20 most habitable exoplanets:")
    top = df.nlargest(20, 'habitability_score')[
        ['pl_name', 'habitability_score', 'esi', 'in_hz', 'pl_eqt', 'pl_rade', 'pl_bmasse', 'is_rocky']
    ]
    print(top.to_string(index=False))

    return df


if __name__ == "__main__":
    run_feature_engineering()
