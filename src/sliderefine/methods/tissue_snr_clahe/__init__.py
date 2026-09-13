"""Thin adapter: the imported algorithm remains untouched in tsclahe."""
from tsclahe import Config, TissueSNRCLAHE
from tsclahe.streaming import fit_streaming
__all__ = ["Config", "TissueSNRCLAHE", "fit_streaming"]
