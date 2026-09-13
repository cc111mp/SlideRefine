# Third-party and provenance policy

The preserved src/tsclahe backend is the user-provided reviewed v0.2.0 artifact, not official IA-CLAHE paper code. Its original license notice is retained. The new workspace is independently authored. A public-release license has not been selected; publication of source is not a grant of a specific license.

NumPy, SciPy, tifffile and Pillow are package dependencies, not vendored implementations. Optional PyTorch is installed only for learning. Retain applicable dependency licenses when redistributing packaged environments.

HiFiEM, visual-prior HE, multiscale redistribution, and Simple Tone Curves are research targets only in this bootstrap. No author code or paper figures are vendored. Source locations in third_party/sources.json are research pointers, NOT dependency locks or verified reuse permission. Pin a source revision and review its license and transitive dependencies before implementation/vendor import.

The repository contains synthetic fixture generators but no specimen images, reference acquisitions, or microscopy-trained weights.
