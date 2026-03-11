# ExoTwin: Digital Twin for Exoplanet Habitability

A digital twin framework that predicts the probability of microbial life on exoplanets based on observable planetary and stellar parameters. Built for the [ENHANCE HACK-4-SAGES 2026](https://hack-4-sages.org/) hackathon (Category C: Digital Twins in Exoplanet Habitability).

## Research Question

> Given observable parameters of an exoplanet, can a digital twin model predict the probability of conditions suitable for microbial life, and identify which factors contribute most to habitability?

## What is ExoTwin?

ExoTwin is a virtual representation of an exoplanet that:

1. **Integrates real observational data** from the NASA Exoplanet Archive (5700+ confirmed exoplanets) and the PHL Habitable Worlds Catalog (expert-curated habitability labels for 70 candidates)
2. **Computes physical properties** — habitable zone boundaries, Earth Similarity Index, surface density, escape velocity, stellar flux
3. **Predicts habitability** using a machine learning model trained on derived habitability scores, cross-validated against PHL expert labels
4. **Enables what-if exploration** — change any parameter in real time and observe how habitability changes
5. **Explains its reasoning** via SHAP values — not just *what* the prediction is, but *why*

## Architecture

```
NASA Exoplanet Archive ─┐
                        ├─► Data Layer ──► Physics Engine ──► ML Prediction ──► What-If Simulation
PHL Habitable Worlds ───┘       │                │                  │                    │
                          merge + fill     ESI, HZ, density   Random Forest       Interactive UI
                          ESI values       escape velocity     + Grad. Boost.      (Streamlit)
                          expert labels    stellar flux        SHAP explainer      Radar / Gauge
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/exotwin.git
cd exotwin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/streamlit_app.py

# Or explore the Jupyter notebook
jupyter notebook notebooks/ExoTwin_Documentation.ipynb
```

## Project Structure

```
exotwin/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # Raw NASA Exoplanet Archive data
│   └── processed/                  # Feature-engineered dataset
├── notebooks/
│   └── ExoTwin_Documentation.ipynb # Full documentation notebook
├── src/
│   ├── data_loader.py              # NASA data fetching and cleaning
│   ├── phl_loader.py               # PHL catalog merge and ESI gap-filling
│   ├── features.py                 # Feature engineering (ESI, HZ, etc.)
│   ├── model.py                    # Model training and evaluation
│   └── digital_twin.py            # ExoplanetTwin class
├── app/
│   └── streamlit_app.py           # Interactive dashboard
├── models/
│   └── best_model.pkl             # Trained model artifact
└── assets/
    └── architecture_diagram.png   # Digital twin architecture diagram
```

## Data Sources

- **NASA Exoplanet Archive** — https://exoplanetarchive.ipac.caltech.edu/ (primary, 6138 confirmed exoplanets via TAP API)
- **PHL Habitable Worlds Catalog** — https://phl.upr.edu/hwc (5599 planets with expert-curated habitability labels: 29 conservative + 41 optimistic candidates, ESI for 5358 planets). Our computed ESI correlates 0.89 with PHL's independent values.

## Key Features Used

| Parameter | Description | Unit |
|-----------|-------------|------|
| `pl_bmasse` | Planet mass | Earth masses |
| `pl_rade` | Planet radius | Earth radii |
| `pl_orbper` | Orbital period | days |
| `pl_orbsmax` | Semi-major axis | AU |
| `pl_orbeccen` | Orbital eccentricity | — |
| `pl_eqt` | Equilibrium temperature | K |
| `st_teff` | Stellar effective temperature | K |
| `st_lum` | Stellar luminosity | log(Solar) |
| `st_mass` | Stellar mass | Solar masses |
| `st_rad` | Stellar radius | Solar radii |

**Engineered features:** Earth Similarity Index (ESI), habitable zone membership, bulk density, escape velocity, stellar flux, rocky planet flag.

## Planet Classification

Exoplanets in our dataset fall into several categories relevant to habitability:

| Category | Criteria | Count | Habitability relevance |
|----------|----------|-------|----------------------|
| Rocky (Terran) | R < 1.6 R⊕ or ρ > 3.5 g/cm³ | ~1500 | Most likely to support surface life |
| Super-Earths | 1.2–2.0 R⊕, 2–10 M⊕ | ~800 | May be rocky or transitional |
| Mini-Neptunes | 2.0–4.0 R⊕ | ~1200 | Thick H/He envelope, less likely habitable on surface |
| Gas Giants | R > 4 R⊕, M > 50 M⊕ | ~2000 | No solid surface, not habitable (moons may be) |
| Tidally locked | P < ~30 d, a < ~0.1 AU, M-dwarf host | Many HZ candidates | One side permanently faces star; extreme temperature contrasts, but terminator zone may be habitable |

**Note on tidal locking:** Neither NASA nor PHL datasets include a direct tidal locking flag. However, most habitable zone candidates around M-dwarf stars (e.g., TRAPPIST-1 e/f/g, Proxima Centauri b, Teegarden's Star b) are almost certainly tidally locked due to their close orbits (periods of days). This is a known limitation — tidal locking affects atmospheric circulation, heat redistribution, and ultimately habitability, but cannot be directly measured from current observations.

## Methodology

1. Data collection from NASA Exoplanet Archive via TAP API (6138 planets)
2. Cross-reference with PHL Habitable Worlds Catalog (expert labels, ESI gap-filling: 1211 → 5582 planets with ESI)
3. Feature engineering based on established astrophysical models (Kopparapu 2013, Schulze-Makuch 2011)
4. Habitability score derivation (composite of HZ membership, ESI, atmosphere retention, composition)
5. Model training: Random Forest + Gradient Boosting with cross-validation
6. Explainability: SHAP TreeExplainer for feature attribution
7. Validation against Solar System bodies and PHL expert-curated habitable candidates (70 planets)
8. Interactive digital twin dashboard with what-if simulation

## References

1. Kopparapu, R. K. et al. (2013). "Habitable Zones around Main-Sequence Stars: New Estimates." *ApJ*, 765(2), 131.
2. Schulze-Makuch, D. et al. (2011). "A Two-Tiered Approach to Assessing the Habitability of Exoplanets." *Astrobiology*, 11(10), 1041–1052.
3. Affholder, A. et al. (2021). "Bayesian analysis of Enceladus's plume data to assess methanogenesis." *Nature Astronomy*, 5, 805–814.
4. Meadows, V. S. & Barnes, R. K. (2018). "Factors Affecting Exoplanet Habitability." In *Handbook of Exoplanets*.
5. Méndez, A. et al. (2021). "Habitable Worlds Catalog." Planetary Habitability Laboratory, UPR Arecibo. https://phl.upr.edu/hwc

## License

MIT

## Team

Built by Team ORION during the ENHANCE HACK-4-SAGES Hackathon, March 9–13, 2026.
