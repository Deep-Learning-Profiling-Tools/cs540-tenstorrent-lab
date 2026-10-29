"""Run: python check_references.py. Requires NumPy; no device needed."""
import numpy as np
from reference import make_mask, ragged_sum_reference, ragged_softmax_reference, case_lengths

rng = np.random.default_rng(540)
lengths = np.array([0, 1, 31, 32, 33, 63, 64, 65])
x = rng.uniform(-80, 80, (8, 96)).astype(np.float32)
mask = make_mask(lengths, 96)
s = ragged_sum_reference(x, lengths)
expected = np.array([np.sum(row[:n], dtype=np.float32) for row, n in zip(x, lengths)])[:,None]
np.testing.assert_allclose(s, expected, atol=2e-4, rtol=2e-5)
y = ragged_softmax_reference(x, lengths)
assert np.isfinite(y).all() and np.all(y >= 0)
assert np.all(y[~mask] == 0)
np.testing.assert_allclose(y.sum(-1), (lengths > 0).astype(np.float32), atol=2e-6)
assert y[1, 0] == 1
equal = ragged_softmax_reference(np.full_like(x, 80), lengths)
for i,n in enumerate(lengths):
    if n: np.testing.assert_allclose(equal[i,:n], 1/n, atol=1e-7)
assert int(case_lengths("balanced").sum()) == int(case_lengths("skewed").sum()) == 555008
assert int(case_lengths("short").sum()) == int(case_lengths("long").sum()) == 524288
large = make_mask(np.array([8191,8192]),8192)
assert large[0].sum() == 8191 and large[1].sum() == 8192
print("CPU reference checks passed: empty/singleton/boundary rows, masking, normalization, equal logits, and distribution counts.")
