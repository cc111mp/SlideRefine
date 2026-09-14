# Third-party notices

## Existing project source and dependencies

The preserved `src/tsclahe` code is the user's supplied reviewed v0.2 project. It is
independent of the original IA-CLAHE authors' code. No publication license for the
project's original code has been selected; see LICENSE_NOTICE.md.

NumPy, SciPy, tifffile, Pillow, pytest and optional Torch/codecs are external package
dependencies and retain their own terms. They are not vendored here.

## HiFiEM contrast excerpt (NEW)

Source: https://github.com/image-infomatics/hifiem
Revision: `477efb57dc2b2d24e1ad56d1044cf99d22d39957`.
Source file: `python/fltemd.py`, Git blob `885114c73e2c04224750340508856c5f2b07dfef`.
License declared by upstream: Apache License 2.0.

A selected `adjust_histogram_v_3_02` computational body with minimal convolution shim
is vendored under `src/sliderefine/methods/hifiem/_vendor/`. Its full LICENSE and NOTICE
are included next to the source, including in the installed wheel. The source has a
shortened docstring/import set and does not include the complete upstream library.
A separately documented AF interface reuses the contrast formulas. No full HiFiEM
restoration, cluster pipeline, example specimen, or notebook is bundled.

Do not remove those notices when redistributing this subset. These terms do not
assign a license to unrelated original SlideRefine code.

## Paper-based code and pending sources

Simple Tone Curves is an independently written implementation of the cited discrete
optimization, not copied author software. Its paper is cited in docs/references.
No article text, figures or PDFs are vendored.

Visual-prior HE and multiscale redistribution remain literature pointers only.
No author code has been copied for them and their algorithm implementations remain
unavailable. See third_party/sources.json for source-specific provenance and scope.
