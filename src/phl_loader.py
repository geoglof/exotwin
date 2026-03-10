"""
Load and merge PHL Habitable Worlds Catalog data with our NASA-based dataset.

The PHL catalog (phl.upr.edu/hwc) provides:
- P_HABITABLE: expert-curated habitability label (0=no, 1=conservative, 2=optimistic)
- P_ESI: Earth Similarity Index computed by PHL
- P_FLUX: stellar flux in Earth fluxes
- P_TYPE: planet classification (Terran, Superterran, Neptunian, Jovian, ...)
- P_HABZONE_CON / P_HABZONE_OPT: habitable zone flags
"""

import os
import pandas as pd
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_phl_catalog():
    path = os.path.join(DATA_DIR, 'raw', 'phl_hwc_full.csv')
    df = pd.read_csv(path)
    return df


def normalize_name(name):
    """Normalize planet names for matching across catalogs."""
    if pd.isna(name):
        return ""
    return (name.strip()
            .replace("'", "'")
            .replace("\u2019", "'")
            .lower()
            .replace("  ", " "))


def merge_phl_with_nasa(nasa_path=None, output_path=None):
    """Merge PHL labels and ESI into our NASA-based feature dataset."""
    if nasa_path is None:
        nasa_path = os.path.join(DATA_DIR, 'processed', 'exoplanets_features.csv')
    if output_path is None:
        output_path = os.path.join(DATA_DIR, 'processed', 'exoplanets_final.csv')

    nasa = pd.read_csv(nasa_path)
    phl = load_phl_catalog()

    nasa['_name_norm'] = nasa['pl_name'].apply(normalize_name)
    phl['_name_norm'] = phl['P_NAME'].apply(normalize_name)

    phl_subset = phl[['_name_norm', 'P_ESI', 'P_HABITABLE', 'P_HABZONE_CON',
                       'P_HABZONE_OPT', 'P_FLUX', 'P_TYPE', 'P_TYPE_TEMP']].copy()
    phl_subset.columns = ['_name_norm', 'phl_esi', 'phl_habitable', 'phl_hz_con',
                          'phl_hz_opt', 'phl_flux', 'phl_type', 'phl_type_temp']

    merged = nasa.merge(phl_subset, on='_name_norm', how='left')
    merged.drop(columns=['_name_norm'], inplace=True)

    matched = merged['phl_esi'].notna().sum()
    total_hab = (merged['phl_habitable'] > 0).sum()

    print(f"Merged PHL data into NASA dataset:")
    print(f"  NASA planets: {len(nasa)}")
    print(f"  PHL matches:  {matched} ({100*matched/len(nasa):.1f}%)")
    print(f"  PHL habitable (conservative): {(merged['phl_habitable']==1).sum()}")
    print(f"  PHL habitable (optimistic):   {(merged['phl_habitable']==2).sum()}")
    print(f"  Not matched: {len(nasa) - matched}")

    # Compare our ESI with PHL's ESI where both exist
    both_esi = merged.dropna(subset=['esi', 'phl_esi'])
    if len(both_esi) > 0:
        corr = both_esi['esi'].corr(both_esi['phl_esi'])
        mae = (both_esi['esi'] - both_esi['phl_esi']).abs().mean()
        print(f"\n  ESI comparison (ours vs PHL) on {len(both_esi)} planets:")
        print(f"    Correlation: {corr:.4f}")
        print(f"    Mean absolute error: {mae:.4f}")

    # Use PHL ESI to fill gaps in our computed ESI
    fill_mask = merged['esi'].isna() & merged['phl_esi'].notna()
    merged.loc[fill_mask, 'esi'] = merged.loc[fill_mask, 'phl_esi']
    print(f"\n  Filled {fill_mask.sum()} missing ESI values from PHL")
    print(f"  Total planets with ESI now: {merged['esi'].notna().sum()}")

    merged.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    return merged


if __name__ == "__main__":
    merge_phl_with_nasa()
