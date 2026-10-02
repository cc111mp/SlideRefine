# Method implementation status — SlideRefine 0.2.0

| ID | Runnable implementation | Provenance and restrictions | Validation |
|---|---|---|---|
| normalization_only | Shared slide-window baseline | Independent project baseline | Synthetic tests |
| tissue_snr_clahe | Preserved tsclahe 0.2 plus WSI adapter | Independent IA-CLAHE-inspired method; default RAM reference or optional disk-backed heuristic cell blocks | Preserved tests, synthetic block/chunk parity and measured RSS; see DISK_STATE_VALIDATION.md |
| hifiem | **Local/global contrast stage only**, explicit `variant="contrast_af"` | Pinned author-code contrast excerpt plus separately documented AF adapter; NOT full HiFiEM, destriping, denoising, or background-model fitting | Tissue-output comparison with pinned excerpt, independent formula, halo/chunk tests |
| simple_tone_curves | Discrete constrained curve fitting and fixed-curve tile application | Independent Python implementation of Bennett/Finlayson Sec. 3.3; explicit supplied target; NOT automatic AF target selection or the full photographic experiment | Hand-checkable optimum, independent small-QP oracle, constraints, serialization and chunk tests |
| multiscale_redistribution | **Unavailable** | Full defining equations/author implementation not verified in this integration | None |
| visual_prior_he | **Unavailable** | Full optimizer/prior/source verification pending | None |

The last two IDs raise `MethodUnavailableError` before output creation. A link or
folder is not a completed algorithm. No generic substitute is labeled as those papers.
`hifiem` requires an explicit supported variant; unsupported stages are not silently approximated.

### CLAHE training status

The default `tissue_snr_clahe` path uses fixed heuristic rules; it does not require
enhancer training. An optional custom controller and paired-image trainer exist,
but no pretrained AF controller or established AF reference-target policy is
provided. Disk-backed fitting supports only the heuristic. The paper's learned
IA-CLAHE is not implemented as a faithful baseline. See [paper supervision](PAPER_RELATIONSHIP.md#training-supervision-in-the-paper)
and [custom training](TRAINING.md). Retraining MIL after heuristic preprocessing
must not be reported as training or evaluating either learned enhancer.

### Tone-curve fitting status

`simple_tone_curves` uses numerical curve fitting with an explicit target vector,
not neural-network training. The core solver does not generate that target.
Athena's separate AF wrapper constructs a patient-balanced reference distribution
from training patients, then fits each slide's curve against the shared reference.
It uses no paired ideal AF target and does not establish a biological reference
standard. See [target provenance and AF policy](references/simple_tone_curves.md).

New reference implementations and equation-to-code pointers:

- [HiFiEM contrast](references/hifiem.md)
- [Simple Tone Curves](references/simple_tone_curves.md)
- [Multiscale source gap](references/multiscale_redistribution.md)
- [Integration validation](REFERENCE_METHODS_VALIDATION.md)

All image experiments in this repository are synthetic. No real-AF improvement,
quantitative fluorescence preservation, or gigapixel throughput is established.
