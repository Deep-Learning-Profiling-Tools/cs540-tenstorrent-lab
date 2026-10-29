# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CS540 adaptation of TT-Lang's single_node_matmul.py.

Kernel only: the course harness supplies tiled BF16 device tensors a, b, out.
All dimensions must be positive multiples of 32. This adaptation has been
source-reviewed, but has not been compiled or run on course hardware.
Upstream: tenstorrent/tt-lang; source details are in provenance.json.
Changes: remove host allocation code and runtime-specific type annotations;
rename operation, use explicit 32-element tiles, check every output dimension.
"""
import ttl


@ttl.operation(grid=(1, 1))
def matmul(a, b, out):
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    assert M > 0 and K > 0 and N > 0
    assert K == b.shape[0]
    assert out.shape[0] == M and out.shape[1] == N
    assert M % 32 == 0 and K % 32 == 0 and N % 32 == 0
    Mt, Kt, Nt = M // 32, K // 32, N // 32

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(a[m, k], a_blk)
                        tx_b = ttl.copy(b[k, n], b_blk)
                        tx_a.wait()
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with out_dfb.reserve() as out_blk:
                    acc = ttl.block.fill(0, shape=out_blk.shape)
                    for _ in range(Kt):
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            acc += a_blk @ b_blk
                    out_blk.store(acc)

    @ttl.datamovement()
    def writer():
        for m in range(Mt):
            for n in range(Nt):
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out[m, n]).wait()
