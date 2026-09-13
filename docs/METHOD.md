# Method specification — v0.2.0

This document specifies the delivered `tsclahe` code, not an implementation of all equations
in the original IA-CLAHE paper. See `PAPER_RELATIONSHIP.md` for that distinction.

## 1. Separate calibration, normalization and enhancement

With a raw dark reference D and a raw uniform-field reference F:

\[
J(r)=(I(r)-D(r))\,\mathrm{median}(F-D)/(F(r)-D(r)).
\]

Nonpositive flat-minus-dark values are rejected. References are optional; the method does
not estimate an illumination field from tissue. Calibration and normalization subtraction
use float64; normalized values are then float32. Integer counts beyond 2**53 are rejected.

With a supplied or heuristically inferred foreground mask M, choose exact tissue percentiles
l=P0.5, h=P99.5, or explicit global/reference bounds. Apply:

\[
x(r)=\operatorname{clip}((J(r)-l)/(h-l),0,1).
\]

Empty or constant-tissue inputs use a recorded conservative fallback and bypass local
enhancement. Nonconstant tissue percentiles are not estimated on a regular stride grid.
This convenience implementation may require substantial memory; streaming expects fixed bounds.

The outputs preserve a separate normalized branch. Neither branch represents raw photon counts.

## 2. Tissue-only tile histograms

Tile size is a pixel count (default 128 by 128), anchored at global origin (0,0). Edge tiles
may be smaller. Spatial interpolation uses actual pixel centers of those edge tiles.

\[
p_i(b)=\frac{\sum_{r\in T_i} M(r)\mathbf1[\mathrm{bin}(x(r))=b]}{\sum_{r\in T_i}M(r)}.
\]

Empty tiles receive a uniform placeholder and zero validity. The histogram has K bins over
[0,1], with 1 in the last bin. LUTs are continuous piecewise-linear on K+1 bin edges. Outputs
are floating point and not restricted to K intensity levels.

## 3. Noise and structural variation proxy

The noise estimate uses fully foreground 2x2 blocks:

\[
d=(x_{00}-x_{01}-x_{10}+x_{11})/2,
\quad\hat\sigma_n=\mathrm{median}(|d-\mathrm{median}(d)|)/0.6744897501960817.
\]

The factor is justified for independent Gaussian noise because the filter has unit squared
coefficient norm. Poisson, correlated, resampled and structured scanner noise violate this model.
Real fine tissue texture can contaminate it.

Let q=P90-P10 on foreground. The implemented structural proxy is:

\[
\sigma_q=q/2.5631031310892007,\quad
\hat\sigma_s=\sqrt{\max(\sigma_q^2-\hat\sigma_n^2,0)},\quad
r_i=\hat\sigma_s/(\hat\sigma_n+\epsilon_n).
\]

This is the archived tsclahe variant's robust-span approximation, **not** the Gaussian-blur
variance estimator discussed for a different source variant. It is not calibrated photon SNR
or a probability that the variation is biologically meaningful.

Validity requires adequate tissue count, tissue fraction, fully masked noise samples, and
nonzero intensity span. A valid tile can still contain artifacts.

## 4. Directional uncertainty and sensor state

The HH filter annihilates additive row/column components. v0.2 adds a separate diagnostic.
Center masked intensities v=x-mean_M(x), and compute total energy E=sum_M(v^2). For each row,
compute its masked sum s_j and count n_j; compute the analogous column quantities. Define:

\[
a_i=\operatorname{clip}\left(\max\left[\sum_j s_{row,j}^2/n_{row,j},
\sum_j s_{col,j}^2/n_{col,j}\right]/E,0,1\right).
\]

Empty groups contribute zero. Constant tiles get score zero but fail contrast validity.
The score is the energy explained by a row OR column mean component. Near-one values are
consistent with directional artifacts, but also with real oriented tissue and some mask shapes.

With smoothstep S(t)=t^2(3-2t), t clipped to [0,1], the conservative directional gate is:

\[
g_A=1-S((a_i-a_{low})/(a_{high}-a_{low})).
\]

Default thresholds are 0.80 and 0.95, engineering settings without microscopy calibration.
`artifact_policy="warn"` sets g_A=1 and retains the diagnostic. This feature does not detect
all oblique stripes, grids, motion, shading, or spectral artifacts.

`upper_window_clip` and `lower_window_clip` count normalized endpoints. They do not estimate
physical saturation and are not used as a surrogate sensor-saturation gate.
When acquisition-specific `sensor_max` is supplied, a saturation mask is measured on raw input
BEFORE dark/flat correction. Its tissue fraction s_i gives a separate gate
`g_sensor=clip(1-s_i/max_sensor_saturated_fraction,0,1)`. Unknown sensor state is flagged explicitly
and does not fabricate a zero measured fraction. Normalized/streaming callers must supply a raw
saturation mask/reader when sensor state is configured; corrected values cannot reconstruct it.

## 5. Contrast need and control proposals

Let f_i be tissue fraction and v_i binary validity. Define:

\[
n_i=\operatorname{clip}(1-q_i/q_{target},0,1),
\quad g_S=S((r_i-r_{low})/(r_{high}-r_{low})),
\quad g_T=S((f_i-f_{min})/(f_{full}-f_{min})).
\]

The maximum allowed blending strength is

\[
A_i=A_{max}\,n_i\,g_S\,g_T\,v_i\,g_A\,g_{sensor}.
\]

The heuristic proposes `c_i=c_min+(c_max-c_min)*g_S*n_i` and `alpha_i=A_i`.
This project does not learn or vary tile dimensions spatially.

The optional controller predicts sigmoid outputs u_i,v_i from a tile-grid CNN:

\[
c_i=c_{min}+(c_{max}-c_{min})u_i g_S n_i,
\quad\alpha_i=A_i v_i.
\]

Inputs are square-root full histograms plus the 14 named scalar channels in `FEATURE_NAMES`.
The CNN is 1x1 -> SiLU -> 3x3 -> SiLU -> 1x1, width 32 by default. Its input is a histogram grid,
not the paper's resized luminance image. No pretrained layers are used. External predictor
proposals are clamped to the deterministic bounds before gain limiting.

## 6. Capped continuous redistribution

Unlike the v0.1 archive's one-pass redistribution, final bin mass is bounded:

\[
\hat p_b=\min(p_b+t,c/K),\qquad\sum_b\hat p_b=1,\quad c\ge1.
\]

For sorted ascending p_(j) and m=1,...,K:

\[
t=\max\left(0,\max_m\frac{1-(K-m)c/K-\sum_{j=1}^m p_{(j)}}m\right).
\]

This active-set solution redistributes mass over bins not at the cap. At c=1 the resulting
histogram is uniform and its bin-edge CDF is identity. It is independently specified here;
it is not bitwise OpenCV/scikit-image CLAHE or a claim to implement the paper's redistribution.
NumPy and PyTorch share the formula; gradchecks avoid active-set switching points.

## 7. Per-tile residual and fixed-context gain bound

Build the CDF F_i with exact endpoints F_i(0)=0 and F_i(1)=1. Instead of interpolating alpha
and F separately, pair each strength with its own mapping:

\[
\delta_i(u)=\alpha_i\operatorname{clip}(F_i(u)-u,-\Delta_{max},\Delta_{max}).
\]

Maximum candidate slope is G_i=K max_b(hat p_b). Limit alpha_i to
`min(alpha_i,(G_max-1)/(G_i-1))` when G_i>1. When the residual clamp is inactive,
`d(u+delta_i)/du=1+alpha_i(F_i'-1)`; when active the derivative is 1. Thus local mappings
remain monotone and their fitted slopes do not exceed G_max, up to floating-point tolerance.

Let beta_i(r) be the usual nonnegative spatial weights summing to one. Define:

\[
\Delta(r,u)=\sum_{i\in N(r)}\beta_i(r)\delta_i(u).
\]

Pairing controls BEFORE interpolation avoids cross terms that would result from multiplying
an independently interpolated strength by an independently interpolated LUT.

## 8. Post-interpolation reliability and exact rejection

A tile is eligible iff its final effective strength is positive. Rasterize eligibility on
its full global pixel footprint. Rejected tiles are exact zero, regardless of neighbors.
Within eligible pixels, take distance d to the nearest rejected pixel and taper inward:

\[
R(r)=S(\operatorname{clip}(d(r)/f_{pixels},0,1)),
\]

with R=0 on rejected pixels. If there are no rejected pixels in the required neighborhood,
R=1. At feather=0, use the binary eligibility raster. This is a tile-rejection envelope,
not a learned pixel probability. Soft nonzero tile strengths are still spatially interpolated.

Distance is computed with a global-coordinate halo wider than the taper radius, so arbitrary
application chunks produce the same result as full application. No full-WSI reliability array
is required. The final output is:

\[
y(r)=\begin{cases}
x(r)+R(r)\Delta(r,x(r)),&M(r)=1,\\
x(r),&M(r)=0.
\end{cases}
\]

For fixed masks, coordinates, reliability and LUTs, the slope cap survives convex interpolation.
Absolute change is at most `max_strength * max_delta`. Rejected tiles and non-tissue pixels
are exactly unchanged relative to the normalized baseline in the tested implementation.

These are **fixed-context properties**, not a Lipschitz bound on the complete adaptive
pipeline, an SNR-improvement theorem, or diagnostic safety. Changing image content changes
statistics and eligibility. Spatial mapping changes can introduce visible variation; real
oriented tissue may be over-rejected. A whole noisy region may include mixed, eligible tiles.

## 9. Learning, serialization and provenance

Controller -> clipping -> CDF -> bounded residual -> interpolation -> image loss is piecewise
differentiable. Input histograms, percentile windows, masks, statistics and eligibility raster
are constants outside that gradient path. The current Torch renderer builds the nonlearned
reliability raster through CPU NumPy/SciPy, so it is not an optimized all-GPU implementation.

Paired training and public inference call the same `prepare` and statistics code. Checkpoints
store operator/feature/preprocessing semantics and enforce the saved intensity/mask policy.
The trainer records active tiles and nonzero gradient steps and stops on all-inactive epochs.
`TRAINING.md` defines target units, splits and selection metrics.

Transform loaders validate shapes, finite values, monotonicity, endpoints, final histogram
caps, effective gain, configuration, and operator versions. Old plans/checkpoints must be
refit/retrained. As with other numeric/model archives, only load trusted files; these checks
are not a claim of protection against every malicious file or resource-exhaustion attack.
