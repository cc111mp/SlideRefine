# Method implementation status — SlideRefine 0.2.0

| ID | Runnable implementation | Provenance and restrictions | Validation |
|---|---|---|---|
| normalization_only | Shared slide-window baseline | Independent project baseline | Synthetic tests |
| tissue_snr_clahe | Preserved tsclahe 0.2 plus WSI adapter | Independent IA-CLAHE-inspired method; global LUT state remains in RAM | Preserved tests and synthetic tile parity |
| hifiem | **Local/global contrast stage only**, explicit `variant="contrast_af"` | Pinned author-code contrast excerpt plus separately documented AF adapter; NOT full HiFiEM, destriping, denoising, or background-model fitting | Tissue-output comparison with pinned excerpt, independent formula, halo/chunk tests |
| simple_tone_curves | Discrete constrained curve fitting and fixed-curve tile application | Independent Python implementation of Bennett/Finlayson Sec. 3.3; explicit supplied target; NOT automatic AF target selection or the full photographic experiment | Hand-checkable optimum, independent small-QP oracle, constraints, serialization and chunk tests |
| multiscale_redistribution | **Unavailable** | Full defining equations/author implementation not verified in this integration | None |
| visual_prior_he | **Unavailable** | Full optimizer/prior/source verification pending | None |

The last two IDs raise `MethodUnavailableError` before output creation. A link or
folder is not a completed algorithm. No generic substitute is labeled as those papers.
`hifiem` requires an explicit supported variant; unsupported stages are not silently approximated.

New reference implementations and equation-to-code pointers:
- [HiFiEM contrast](references/hifiem.md)
- [Simple Tone Curves](references/simple_tone_curves.md)
- [Multiscale source gap](references/multiscale_redistribution.md)
- [Integration validation](REFERENCE_METHODS_VALIDATION.md)

All image experiments in this repository are synthetic. No real-AF improvement,
quantitative fluorescence preservation, or gigapixel throughput is established.
