# Simple Tone Curves: discrete optimization reference

Bennett, J.; Finlayson, G. *Simple tone curves: theory and applications*.
The Visual Computer 42, 302 (2026). DOI: `10.1007/s00371-026-04505-y`.
Primary article: https://link.springer.com/article/10.1007/s00371-026-04505-y
Method verified from Sections 3.2 and 3.3, including the PDF constraint matrices.
This is **independent Python code**, not an author implementation or a reproduction
of the photographic preference experiments. No paper PDF/figures are redistributed.

## Implemented problem

Given explicitly supplied, uniformly sampled monotone target values t, fit y by
minimizing `sum((y-t)**2)` under nonnegative first differences, preserved target
endpoints, and one of the four signed-second-difference cases. Cases 1/2 are fully
convex/concave; cases 3/4 permit one curvature sign switch in either direction.
For each case and permitted switch, solve its convex quadratic program; choose the
lowest-error feasible candidate. Squaring the paper's L2 norm preserves its minimizer.

The reference supports 4..256 samples on `linspace(0,1,n)`; the paper uses n=100.
Endpoints equal t[0] and t[-1], not automatically 0 and 1. Input targets must be
nondecreasing and in [0,1]. The routine does not infer a target from tissue appearance.

## Equation-to-code contract

| Paper component | Implementation | Verification |
|---|---|---|
| First differences, Eqs. 12–14 | `constraint_matrix`: D1 | Monotonic fitted samples |
| Curvature cases, Eqs. 15–26 | D2 with case/switch signs | Four zero-error shape fixtures; candidate coverage |
| Target endpoints, Eqs. 27–29 | Eliminate endpoints from optimization variables | Nonzero/nonunit endpoint fixture |
| Search objective, Eq. 30 | `fit_simple_tone_curve` | Closed small example and independent active-face QP oracle |
| Uniform sampling/PCHIP, Sec. 3.2 | `SimpleToneCurve.apply` | Continuous application monotonicity and serialization |

`inflection` uses the paper's ONE-BASED f, not an array index. For case 3, second
differences centered at i=2..f are nonnegative and those at i=f+1..n-1 nonpositive.
Case 4 reverses the signs. f=n-1 duplicates a pure case; duplicate searches are omitted,
not feasible curves. A full nontrivial search solves `2*n-4` distinct QPs.

SciPy SLSQP solves each QP with analytic objective and constraint Jacobians. Redundant
endpoint-range bounds prevent numerical excursions at flat endpoint sections without
changing the feasible set. A solver failure aborts the fit; unsuccessful cases are
not silently skipped. Already feasible targets are exact zero-objective solutions.
These are numerical reference results, not a formal machine-checked optimality proof.

`[0,.1,.5,.6,1]` has fitted samples `[0,.15,.4,.65,1]` and squared error .015 in the
hand-checkable fixture. A separate tiny-problem oracle enumerates active faces using
linear algebra, without calling the production constraint builder or optimizer.

## Interpolation and domain boundary

PCHIP is the default as in the paper; linear interpolation is an explicitly labeled
alternative. The curvature restriction applies to **sampled second differences**.
It does not establish a zero/one-inflection theorem for the continuous PCHIP polynomial.
The implementation does not add a generic sigmoid and label it the paper's optimizer.

This release uses normalized linear grayscale AF intensities. CIELAB conversion,
log-brightness encoding, expert-image target extraction, automatic target prediction,
and reproduction of the paper's datasets/metrics are not included. An AF target policy
is a separate experiment. Never derive a deployment target from held-out reference
images and report it as reference-free inference.

## Execution and reproducibility

Fit a curve once, save its versioned JSON, and reuse it across all slide chunks.
The JSON records the supplied target, solution, case, switch, solver, interpolation,
and objective. Loading validates the numerical invariants. No neighboring pixels are
needed for fixed-curve application. The WSI adapter restores non-tissue pixels to the
baseline and exports validity for missing coverage. Thus monotone ordering guarantees
for a globally applied curve should not be generalized across differently masked areas.

`configs/tone_target.example.json` is a synthetic fixture only. Fitting should be done
before rendering and must not repeat independently at each execution tile or viewport.
