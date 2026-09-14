"""Independent discrete Simple Tone Curves solver (Bennett & Finlayson, 2026).

Implements Sec. 3.3 / Eqs. 12--30: least-squares projection onto the union
of monotone, endpoint-preserving curves with at most one curvature sign switch.
Not author code. No AF target selection or logarithmic/color conversion is implied.
See docs/references/simple_tone_curves.md for interpolation and solver limitations.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

SCHEMA = "sliderefine.simple-tone-curve/v1"
TOL = 1e-7


def _samples(values, name):
    a = np.asarray(values)
    if a.dtype.kind not in "uif" or a.ndim != 1 or not 4 <= a.size <= 256:
        raise ValueError(f"{name} must contain 4..256 real, uniformly spaced samples")
    a = a.astype(np.float64, copy=True)
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return a


def constraint_matrix(n: int, case: int, inflection: int | None = None):
    """Paper's D and R stacked, so M @ curve >= 0.

    `inflection` is the paper's ONE-BASED f, not a Python array index.
    Cases 1/2: convex/concave. Cases 3/4: convex-concave/concave-convex.
    """
    if type(n) is not int or n < 4 or n > 256 or type(case) is not int or case not in (1,2,3,4):
        raise ValueError("Invalid curve size or case")
    d1, d2 = np.diff(np.eye(n), axis=0), np.diff(np.eye(n), n=2, axis=0)
    if case in (1,2):
        if inflection is not None:
            raise ValueError("Cases 1 and 2 do not have an inflection")
        signs = np.ones(n-2) * (1 if case == 1 else -1)
    else:
        if type(inflection) is not int or not 2 <= inflection <= n-1:
            raise ValueError("inflection must be one-based f in [2,n-1]")
        signs = np.ones(n-2)
        signs[inflection-1:] = -1
        if case == 4:
            signs *= -1
    return np.vstack((d1, signs[:, None]*d2))


def _candidates(n):
    yield 1, None
    yield 2, None
    # f=n-1 duplicates cases 1/2 exactly; omit that duplicate, not a feasible set.
    for case in (3,4):
        for f in range(2,n-1):
            yield case, f


@dataclass(frozen=True)
class SimpleToneCurve:
    samples: tuple[float, ...]
    target: tuple[float, ...]
    case: int
    inflection: int | None
    interpolation: str = "pchip"
    solver: str = "scipy-slsqp"
    candidates_solved: int = 0

    def __post_init__(self):
        y, t = _samples(self.samples, "samples"), _samples(self.target, "target")
        if y.shape != t.shape or np.any(np.diff(t) < 0):
            raise ValueError("Target must be nondecreasing and match the fitted sample count")
        if self.interpolation not in ("pchip", "linear"):
            raise ValueError("interpolation must be pchip or linear")
        if type(self.candidates_solved) is not int or self.candidates_solved < 0:
            raise ValueError("Invalid candidate count")
        if abs(y[0]-t[0]) > TOL or abs(y[-1]-t[-1]) > TOL:
            raise ValueError("Fitted endpoints must equal the target endpoints")
        if np.min(constraint_matrix(y.size, self.case, self.inflection) @ y) < -TOL:
            raise ValueError("Fitted curve violates monotonicity or discrete curvature constraints")
        object.__setattr__(self, "samples", tuple(map(float,y)))
        object.__setattr__(self, "target", tuple(map(float,t)))

    @property
    def squared_error(self):
        return float(np.sum((np.asarray(self.samples)-self.target)**2))

    def apply(self, image):
        a = np.asarray(image)
        if a.dtype.kind not in "uif" or not np.isfinite(a).all():
            raise ValueError("Image must contain finite real normalized values")
        if a.size and (a.min() < 0 or a.max() > 1):
            raise ValueError("Tone-curve input must already be normalized to [0,1]")
        x = np.linspace(0,1,len(self.samples))
        if self.interpolation == "pchip":
            out = PchipInterpolator(x, self.samples, extrapolate=False)(a)
        else:
            out = np.interp(a, x, self.samples)
        return np.clip(out,0,1).astype(np.float32)

    def to_dict(self):
        return dict(schema=SCHEMA, samples=list(self.samples), target=list(self.target),
                    case=self.case, inflection=self.inflection, interpolation=self.interpolation,
                    solver=self.solver, candidates_solved=self.candidates_solved,
                    squared_error=self.squared_error, domain="normalized-linear-[0,1]")

    def save(self, path):
        # No overwrite: a curve is fitted state, not a mutable run-wide parameter.
        with Path(path).open("x", encoding="utf-8") as f:
            json.dump(self.to_dict(),f,indent=2,allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        required = {"schema","samples","target","case","inflection","interpolation","solver",
                    "candidates_solved","squared_error","domain"}
        if not isinstance(data,dict) or set(data) != required or data["schema"] != SCHEMA:
            raise ValueError("Unsupported or incomplete tone-curve schema")
        if data["domain"] != "normalized-linear-[0,1]":
            raise ValueError("Unsupported tone-curve domain")
        obj = cls(**{k:data[k] for k in required - {"schema","squared_error","domain"}})
        if not np.isfinite(data["squared_error"]) or not np.isclose(
                obj.squared_error, data["squared_error"], rtol=1e-8, atol=1e-12):
            raise ValueError("Stored objective does not match the curve and target")
        return obj

    @classmethod
    def load(cls, path):
        if Path(path).stat().st_size > 100_000:
            raise ValueError("Tone curve file exceeds the size limit")
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_simple_tone_curve(target, *, interpolation="pchip", maxiter=1000, ftol=1e-12):
    """Search ALL distinct discrete shape cases; fail on an unsuccessful QP.

    Values are sampled at linspace(0,1,n). Endpoints are fixed to the supplied
    target, not forced to 0/1. This optimizes a supplied tone curve, not an image.
    The squared L2 objective has the same minimizer as the paper's L2 norm.
    PCHIP follows the paper; the sign-switch guarantee applies to sampled second
    differences, NOT a theorem about the continuous cubic interpolant's curvature.
    """
    t = _samples(target, "target")
    if np.any(np.diff(t) < 0):
        raise ValueError("Target tone curve must be nondecreasing")
    if interpolation not in ("pchip", "linear"):
        raise ValueError("interpolation must be pchip or linear")
    if type(maxiter) is not int or maxiter < 1 or not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("Invalid solver limits")
    cases = list(_candidates(t.size))
    # Feasibility + zero objective proves optimality; no numerical search needed.
    for case, f in cases:
        if np.min(constraint_matrix(t.size,case,f) @ t) >= -1e-13:
            return SimpleToneCurve(tuple(t),tuple(t),case,f,interpolation,"analytic-zero-error",0)
    initial = np.linspace(t[0],t[-1],t.size)[1:-1]
    best, best_error = None, np.inf
    for case, f in cases:
        matrix = constraint_matrix(t.size,case,f)
        a = matrix[:,1:-1]
        offset = matrix[:,0]*t[0] + matrix[:,-1]*t[-1]
        result = minimize(lambda z: 0.5*np.sum((z-t[1:-1])**2), initial,
                          jac=lambda z: z-t[1:-1], method="SLSQP",
                          # Redundant with monotonicity/endpoints, but stops floating
                          # roundoff just outside [0,1] at flat endpoint sections.
                          bounds=[(float(t[0]),float(t[-1]))]*(t.size-2),
                          constraints={"type":"ineq", "fun":lambda z: a@z+offset,
                                       "jac":lambda z: a},
                          options={"maxiter":maxiter,"ftol":ftol})
        y = np.r_[t[0], result.x, t[-1]]
        if not result.success or not np.isfinite(y).all() or np.min(matrix@y) < -TOL:
            raise RuntimeError(f"Tone QP failed for case={case}, f={f}: {result.message}")
        error = float(np.sum((y-t)**2))
        if error < best_error:
            best, best_error = (y.copy(),case,f), error
    y, case, f = best
    return SimpleToneCurve(tuple(y),tuple(t),case,f,interpolation,"scipy-slsqp",len(cases))
