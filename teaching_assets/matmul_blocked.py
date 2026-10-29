# SPDX-FileCopyrightText: (c) 2025-2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CS540 blocked/multicore adaptation of the upstream TT-Lang matmul examples.

Kernel only, source-reviewed and not hardware-validated. Padding to full block
boundaries belongs to the harness. Defaults use 2x2 output tiles and K blocks
of two tiles. The actual selected compiler strategy determines accumulation
storage and numerical behavior; inspect the generated IR.
"""
import ttl

BM, BN, BK = 2, 2, 2


@ttl.operation(grid="full")
def matmul_blocked(a, b, out):
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    assert M > 0 and K > 0 and N > 0
    assert K == b.shape[0]
    assert out.shape[0] == M and out.shape[1] == N
    assert M % (32 * BM) == 0 and N % (32 * BN) == 0
    assert K % (32 * BK) == 0
    mb, nb, kb = M // (32 * BM), N // (32 * BN), K // (32 * BK)
    cores = ttl.grid_size(dims=1)
    per_core = (mb * nb + cores - 1) // cores
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(BM, BK), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(BK, BN), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(BM, BN), block_count=2)

    @ttl.datamovement()
    def reader():
        node = ttl.node(dims=1)
        for local in range(per_core):
            q = node * per_core + local
            if q < mb * nb:
                m, n = q // nb, q % nb
                for k in range(kb):
                    with a_dfb.reserve() as aa, b_dfb.reserve() as bb:
                        ta = ttl.copy(a[m*BM:(m+1)*BM, k*BK:(k+1)*BK], aa)
                        tb = ttl.copy(b[k*BK:(k+1)*BK, n*BN:(n+1)*BN], bb)
                        ta.wait()
                        tb.wait()

    @ttl.compute()
    def compute():
        node = ttl.node(dims=1)
        for local in range(per_core):
            q = node * per_core + local
            if q < mb * nb:
                with out_dfb.reserve() as cc:
                    acc = ttl.block.fill(0, shape=cc.shape)
                    for _ in range(kb):
                        with a_dfb.wait() as aa, b_dfb.wait() as bb:
                            acc += aa @ bb
                    cc.store(acc)

    @ttl.datamovement()
    def writer():
        node = ttl.node(dims=1)
        for local in range(per_core):
            q = node * per_core + local
            if q < mb * nb:
                m, n = q // nb, q % nb
                with out_dfb.wait() as cc:
                    ttl.copy(cc, out[m*BM:(m+1)*BM, n*BN:(n+1)*BN]).wait()
