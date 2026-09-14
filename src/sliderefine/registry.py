"""No silent substitutions for unimplemented paper methods; no eager Torch import."""
from dataclasses import asdict, dataclass

class MethodUnavailableError(NotImplementedError):
    pass

@dataclass(frozen=True)
class MethodInfo:
    id: str
    status: str
    provenance: str
    validation: str
    note: str

_METHODS = (
    MethodInfo("normalization_only", "implemented", "project baseline", "synthetic tests",
               "One fixed or slide-fitted window; not a research-paper implementation."),
    MethodInfo("tissue_snr_clahe", "implemented", "independent IA-CLAHE-inspired extension", "synthetic tests",
               "Preserved tsclahe v0.2 backend; LUT state remains in RAM, no giant-WSI benchmark."),
    MethodInfo("hifiem", "implemented", "pinned upstream contrast excerpt + AF adapter", "synthetic tests",
               "ONLY contrast_af variant; no full HiFiEM/stripe/denoise pipeline. Explicit config required."),
    MethodInfo("visual_prior_he", "planned", "paper implementation pending", "not evaluated",
               "Verify equations, solver, brightness prior, and code availability."),
    MethodInfo("multiscale_redistribution", "planned", "paper implementation pending", "not evaluated",
               "Preserve per-scale redistribution and global slide grids."),
    MethodInfo("simple_tone_curves", "implemented", "independent discrete QP reimplementation", "synthetic tests",
               "Sec. 3.3 shape-constrained curve fitting; explicit fitted target curve required."),
)

def available_methods() -> list[dict]:
    return [asdict(m) for m in _METHODS]

def require_method(method_id: str) -> MethodInfo:
    for method in _METHODS:
        if method.id == method_id:
            if method.status != "implemented":
                raise MethodUnavailableError(f"{method_id} is planned, not implemented. {method.note}")
            return method
    raise ValueError(f"Unknown method {method_id!r}; run 'sliderefine methods'")
