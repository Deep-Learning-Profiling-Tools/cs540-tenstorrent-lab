"""CPU definitions for FP32 exercise inputs and outputs.
NumPy alone does not supply the course device runtime. These are reference
semantics, not implementations eligible for OJ submission.
"""
import numpy as np

def make_mask(lengths, width):
    lengths = np.asarray(lengths, dtype=np.int64)
    if width < 1 or np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError("lengths must be between zero and the positive width")
    return np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]

def ragged_sum_reference(x, lengths):
    x = np.asarray(x, dtype=np.float32)
    mask = make_mask(lengths, x.shape[1])
    return np.where(mask, x, 0).sum(axis=-1, keepdims=True, dtype=np.float32)

def ragged_softmax_reference(x, lengths):
    x = np.asarray(x, dtype=np.float32)
    mask = make_mask(lengths, x.shape[1])
    if not np.isfinite(x).all():
        raise ValueError("reference requires finite inputs")
    masked = np.where(mask, x, -np.inf)
    nonempty = np.asarray(lengths)[:, None] > 0
    rowmax = np.where(nonempty, masked.max(axis=-1, keepdims=True), 0)
    weights = np.where(mask, np.exp(masked - rowmax), 0)
    denom = np.where(nonempty, weights.sum(axis=-1, keepdims=True), 1)
    return weights / denom

def case_lengths(name):
    if name == "boundaries":
        return np.tile(np.array([0, 1, 31, 32, 33, 63, 64, 65]), 32)
    definitions = {"short": ([4096, 4096], [32, 96]),
                   "balanced": ([1024], [542]),
                   "skewed": ([960, 64], [32, 8192]),
                   "long": ([32], [16384])}
    counts, lengths = definitions[name]
    return np.repeat(np.array(lengths, dtype=np.int64), counts)
