# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CS540 single-core blocked adaptation of matmul.py.

The course runner supplies preallocated tiled BF16 tensors.
With BM=BN=BK=2, scalar dimensions must be multiples of 64.
Device compilation and numerical validation are pending.
"""
import ttl

BM, BN, BK = 2, 2, 2


@ttl.operation(grid=(1, 1))
def matmul_larger(a, b, out):
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    assert M > 0 and K > 0 and N > 0
    assert K == b.shape[0]
    assert out.shape[0] == M and out.shape[1] == N
    assert M % (32 * BM) == 0 and K % (32 * BK) == 0
    assert N % (32 * BN) == 0
    Mt, Kt, Nt = M // (32 * BM), K // (32 * BK), N // (32 * BN)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(BM, BK), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(BK, BN), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(BM, BN), block_count=2)

    @ttl.datamovement()
    def reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(a[m*BM:(m+1)*BM, k*BK:(k+1)*BK], a_blk)
                        tx_b = ttl.copy(b[k*BK:(k+1)*BK, n*BN:(n+1)*BN], b_blk)
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
                    ttl.copy(out_blk, out[m*BM:(m+1)*BM, n*BN:(n+1)*BN]).wait()
