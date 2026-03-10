"""
Download and clean exoplanet data from the NASA Exoplanet Archive.
Uses the TAP (Table Access Protocol) API to fetch the Planetary Systems table.
"""

import os
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLUMNS = [
    "pl_name",          # Planet name
    "hostname",         # Host star name
    "sy_snum",          # Number of stars in system
    "sy_pnum",          # Number of planets in system
    "discoverymethod",  # Discovery method
    "disc_year",        # Discovery year
    "pl_orbper",        # Orbital period (days)
    "pl_orbsmax",       # Semi-major axis (AU)
    "pl_orbeccen",      # Orbital eccentricity
    "pl_bmasse",        # Planet mass (Earth masses)
    "pl_rade",          # Planet radius (Earth radii)
    "pl_dens",          # Planet density (g/cm^3)
    "pl_eqt",           # Equilibrium temperature (K)
    "st_teff",          # Stellar effective temperature (K)
    "st_rad",           # Stellar radius (Solar radii)
    "st_mass",          # Stellar mass (Solar masses)
    "st_lum",           # Stellar luminosity (log10 Solar)
    "st_age",           # Stellar age (Gyr)
    "st_met",           # Stellar metallicity (dex)
    "st_spectype",      # Stellar spectral type
]


def download_exoplanet_data(force=False):
    """Download the full confirmed exoplanets table from NASA."""
    raw_path = os.path.join(RAW_DIR, 'nasa_exoplanets_raw.csv')

    if os.path.exists(raw_path) and not force:
        print(f"Raw data already exists at {raw_path}, skipping download.")
        return pd.read_csv(raw_path)

    query = f"SELECT {','.join(COLUMNS)} FROM ps WHERE default_flag = 1"

    print("Downloading from NASA Exoplanet Archive...")
    response = requests.get(NASA_TAP_URL, params={
        "query": query,
        "format": "csv",
    }, timeout=120)
    response.raise_for_status()

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(raw_path, 'wb') as f:
        f.write(response.content)

    df = pd.read_csv(raw_path)
    print(f"Downloaded {len(df)} exoplanets with {len(df.columns)} columns.")
    return df


def clean_data(df):
    """Basic cleaning: drop duplicates, report missing values."""
    df = df.drop_duplicates(subset=['pl_name'], keep='first')

    print(f"\n--- Dataset Overview ---")
    print(f"Total planets: {len(df)}")
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    for col in COLUMNS:
        if col in df.columns:
            n_miss = missing[col]
            pct = 100 * n_miss / len(df)
            print(f"  {col:20s}: {n_miss:5d} missing ({pct:5.1f}%)")

    return df


def save_processed(df):
    """Save the cleaned dataset."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, 'exoplanets_clean.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved cleaned data to {path}")
    return path


if __name__ == "__main__":
    df = download_exoplanet_data()
    df = clean_data(df)
    save_processed(df)
