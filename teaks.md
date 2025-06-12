The notebook now:

| **Extension**           | **Where to look / what to tweak**                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MRI sweep**           | `TR_vals_ms`, `TE_vals_ms` define the grid. Heat-map shows CNR; the orange × pinpoints the global optimum (≈ 3 s / 40 ms in the default ranges).                                                          |
| **Ultrasound speckle**  | Rayleigh coefficient of variation (`cv_rayleigh`) folds in multiplicative speckle; the shaded band in the depth plot is ±1 σ. The printed CNR reflects both speckle and electronic noise.                 |
| **X-ray polychromatic** | `tungsten_brems_spectrum()` + 1 cm Al bow-tie filter → filtered spectrum, then path-length integration over energies. You can alter `filter_thickness_cm` or swap in a more sophisticated spectral model. |

Feel free to rerun cells after adjusting any of the hyper-parameters (frequency, filter thickness, TR range, etc.) to explore different scenarios. Let me know if you’d like further refinements—e.g. variable bow-tie thickness across the field, alternative speckle statistics, or SAR/dose estimates.
