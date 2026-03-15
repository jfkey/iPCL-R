# This module has been removed.
# Embedding-level VQ (VectorQuantizer) has been replaced by
# coordinate-level VQ (CoordinateVQ) in coordinate_vq.py.
#
# CoordinateVQ quantizes raw 3D coordinates BEFORE Fourier encoding,
# which is more effective for spatial generalization than quantizing
# the high-dimensional embedding output.

raise ImportError(
    "flow.models.vq has been removed. "
    "Use flow.models.coordinate_vq.CoordinateVQ instead."
)
