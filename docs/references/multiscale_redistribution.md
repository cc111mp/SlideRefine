# Multiscale redistribution: not implemented

Target paper: *Enhancing Local Contrast in Low-Light Images: A Multiscale Model with
Adaptive Redistribution of Histogram Excess* (2025), DOI `10.3390/math13203282`.
Primary pointer: https://www.mdpi.com/2227-7390/13/20/3282

The full defining method/equations could not be retrieved in this integration attempt.
The available metadata and abstract do not establish the complete redistribution,
scale-fusion and boundary rules. An author implementation and its license have not
been verified. The registry therefore continues to raise `MethodUnavailableError`.

To implement this method, obtain the full paper or a verified author implementation,
record its version, transcribe the redistribution and fusion rules, and test small
histograms plus full/cropped multiscale outputs. Use globally anchored analysis grids
at every scale. Any grayscale AF adaptation must be separated from the reference.
Do not replace this with three calls to ordinary CLAHE or an invented weighted blend
while retaining the paper's name. No invented equations or placeholder outputs are
included here. The source pointer is not a claim of implementation or paper fidelity.
