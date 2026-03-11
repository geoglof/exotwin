# ExoTwin — Video Script (2 minutes)

---

## 0:00–0:15 — Hook + Problem

**[SCREEN: Space animation / JWST exoplanet image]**

**NARRATOR:**
We have discovered over five thousand seven hundred exoplanets — but which of them could harbour life? This is one of the most important questions in modern astrobiology. The problem is we cannot fly to any of them. We need a tool that lets us study their conditions remotely.

---

## 0:15–0:30 — Our Solution

**[SCREEN: ExoTwin logo → digital twin architecture diagram]**

**NARRATOR:**
We present ExoTwin — a digital twin for exoplanets. ExoTwin combines real observational data with physics-based models and machine learning to predict the probability of conditions suitable for microbial life on a given planet. It is an interdisciplinary approach — connecting astronomy, biology, physics, and computer science.

---

## 0:30–0:45 — Methodology

**[SCREEN: Architecture diagram (Data Layer → Physics Engine → ML → What-If)]**

**NARRATOR:**
Our data comes from two sources: the NASA Exoplanet Archive — over six thousand planets — and the PHL Habitable Worlds Catalog, which contains expert habitability assessments for seventy candidates.

From these, we compute nineteen physical features: habitable zone boundaries following Kopparapu, the Earth Similarity Index, atmosphere retention capability, orbital stability, stellar spectral suitability for photosynthesis, and host star stability.

These features feed a Gradient Boosting model that achieves an R-squared of zero point nine nine on the test set.

---

## 0:45–1:15 — Live Demo

**[SCREEN: Streamlit app — Database tab]**

**NARRATOR:**
In the database, we can browse over six thousand planets. We see a table of top candidates, a mass-radius diagram coloured by habitability, a habitable zone plot, and an interactive 3D explorer.

**[SCREEN: Switch to Custom Planet tab]**

But the heart of ExoTwin is the what-if simulation — the core of any digital twin. I can create a custom planet, set its mass, radius, orbit, and star parameters. The model instantly computes a habitability score and shows a radar profile compared to Earth.

**[SCREEN: Switch to What-If tab, select TRAPPIST-1 e]**

In What-If mode, I select a known planet — for example TRAPPIST-1 e — and modify its parameters. I see the original score, the modified score, and the change in real time. The sensitivity analysis shows which parameter has the biggest impact on this specific planet's habitability.

---

## 1:15–1:35 — Results + Validation

**[SCREEN: Validation chart — known planets + SHAP importance]**

**NARRATOR:**
We validated the model on Solar System planets and known exoplanets. Earth scores highest among planets orbiting Sun-like stars, Jupiter scores lowest — as expected.

SHAP analysis reveals which features drive predictions. Density, rocky composition, and equilibrium temperature dominate, but the new features — spectral suitability and stellar stability — rank fifth and sixth. This confirms that the host star type significantly affects the chances for life.

Independent validation on seventy PHL-labelled planets confirms our model correctly distinguishes habitable from non-habitable worlds.

---

## 1:35–1:50 — Limitations + Future

**[SCREEN: Slide with limitations list]**

**NARRATOR:**
We acknowledge key limitations. The transit detection method favours large planets close to their star. For most exoplanets, atmospheric composition remains unknown. And our definition of habitability is based on Earth-like life — alien biochemistry may look entirely different.

In the future, we plan to integrate atmospheric data from the James Webb Space Telescope, and extend the model to subsurface ocean worlds — like Enceladus or Europa.

---

## 1:50–2:00 — Closing

**[SCREEN: ExoTwin logo + GitHub repository link]**

**NARRATOR:**
ExoTwin is fully reproducible — all data is public, the code is open-source, and the notebook runs end-to-end. Our digital twin can help scientists prioritise observation targets for JWST and future missions like the Habitable Worlds Observatory.

ExoTwin — helping humanity find life among the stars.

---

## Video Checklist

- [ ] All jury keywords mentioned: digital twin, habitability, reproducible, interdisciplinary
- [ ] Architecture diagram visible (~5 seconds)
- [ ] Streamlit demo: Database → Custom Planet → What-If
- [ ] SHAP feature importance visible
- [ ] Validation on known planets visible
- [ ] Limitations mentioned
- [ ] JWST / future missions mentioned
- [ ] GitHub link visible
- [ ] Total time <= 2:00

## References

1. Kopparapu, R. K. et al. (2013). "Habitable Zones around Main-Sequence Stars: New Estimates." *The Astrophysical Journal*, 765(2), 131.
2. Schulze-Makuch, D. et al. (2011). "A Two-Tiered Approach to Assessing the Habitability of Exoplanets." *Astrobiology*, 11(10), 1041-1052.
3. Affholder, A. et al. (2021). "Bayesian analysis of Enceladus's plume data to assess methanogenesis." *Nature Astronomy*, 5, 805-814.
4. Meadows, V. S. & Barnes, R. K. (2018). "Factors Affecting Exoplanet Habitability." In *Handbook of Exoplanets*, Springer.
5. NASA Exoplanet Archive. https://exoplanetarchive.ipac.caltech.edu/
6. Méndez, A. et al. (2021). "Habitable Worlds Catalog." Planetary Habitability Laboratory, UPR Arecibo. https://phl.upr.edu/hwc
7. Madhusudhan, N. et al. (2023). "Carbon-bearing molecules in a possible hycean atmosphere." *The Astrophysical Journal Letters*, 956, L13.
8. Bolmont, E. et al. (2016). "Habitability of planets on eccentric orbits: Limits of the mean flux approximation." *Astronomy & Astrophysics*, 591, A106.
9. Lingam, M. & Loeb, A. (2018). "Photosynthesis on habitable planets around low-mass stars." *The Astrophysical Journal*, 846, 21.
10. Yang, J. et al. (2014). "Strong Dependence of the Inner Edge of the Habitable Zone on Planetary Rotation Rate." *ApJ Letters*, 787, L2.
