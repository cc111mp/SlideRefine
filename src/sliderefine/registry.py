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
    MethodInfo("hifiem", "planned", "upstream adapter pending", "not evaluated",
               "Verify/pin author code and licenses; adapt individual EM stages explicitly."),
    MethodInfo("visual_prior_he", "planned", "paper implementation pending", "not evaluated",
               "Verify equations, solver, brightness prior, and code availability."),
    MethodInfo("multiscale_redistribution", "planned", "paper implementation pending", "not evaluated",
               "Preserve per-scale redistribution and global slide grids."),
    MethodInfo("simple_tone_curves", "planned", "paper implementation pending", "not evaluated",
               "Define target-curve policy; a generic gamma curve is not the paper."),
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
