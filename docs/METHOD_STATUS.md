# Method implementation status

| ID | Current implementation | Provenance | Domain validation |
|---|---|---|---|
| normalization_only | Runnable plane/tile baseline | Independent project baseline | Synthetic tests |
| tissue_snr_clahe | Runnable; imported tsclahe v0.2 plus thin WSI adapter | IA-CLAHE-inspired, not official or faithful reproduction | Synthetic tests only |
| hifiem | Planned | Author adapter and stage-specific AF variant pending | None |
| visual_prior_he | Planned | Equations/solver/code provenance verification pending | None |
| multiscale_redistribution | Planned | Paper reimplementation and grayscale AF mapping pending | None |
| simple_tone_curves | Planned | Paper curve constraints and target policy pending | None |

Selecting a planned ID raises MethodUnavailableError before output creation. A folder is not an implementation. A simple gamma/sigmoid baseline must not be labeled Simple Tone Curves paper reproduction. HiFiEM's EM conventions must not silently become AF defaults. Preserve the defining optimizer/redistribution behavior in the HE papers.

Read docs/PAPER_RELATIONSHIP.md for the imported backend's relationship to IA-CLAHE. See third_party/sources.json for literature pointers, not validated software licenses.
