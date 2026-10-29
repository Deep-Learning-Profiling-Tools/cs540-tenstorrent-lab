"""Teaching adaptation of the TT-Lang operation and reduction API.

Source-checked against tt-lang a47484458bd964a6b92c9e4698aeb922585f08cc.
These teaching kernels have not been compiled or executed on Tenstorrent hardware.
The course host harness supplies preallocated tiled device tensors. Do not call
these functions with ordinary Torch tensors. No host allocation API is taught here.

Small-width BF16 baseline: x, valid and out use BF16 tiled storage. x and valid
have padded shape (R_pad, W_pad), with R_pad
and W_pad divisible by 32 and W_pad <= 256 for the initial exercises. All x values
are finite; softmax logits are in [-80, 80]. valid is exactly 0 or 1, marks the
first length[r] entries in each row, and is zero on padded rows and columns.
Sum output has padded shape (R_pad, 32): the logical result is its column zero.
Softmax output has padded shape (R_pad, W_pad). Each operation returns None.
"""

import ttl


@ttl.operation(grid=(1, 1))
def ragged_sum(x, valid, out):
    row_groups = x.shape[0] // 32
    width_tiles = x.shape[1] // 32
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, width_tiles), block_count=2)
    valid_dfb = ttl.make_dataflow_buffer_like(valid, shape=(1, width_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(row_groups):
            with x_dfb.wait() as x_blk, valid_dfb.wait() as valid_blk, out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.reduce_sum(x_blk * valid_blk, dims=[1], shape=(1, 1)))

    @ttl.datamovement()
    def reader():
        for row in range(row_groups):
            with x_dfb.reserve() as x_blk, valid_dfb.reserve() as valid_blk:
                tx_x = ttl.copy(x[row:row + 1, 0:width_tiles], x_blk)
                tx_valid = ttl.copy(valid[row:row + 1, 0:width_tiles], valid_blk)
                tx_x.wait()
                tx_valid.wait()

    @ttl.datamovement()
    def writer():
        for row in range(row_groups):
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[row:row + 1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def ragged_softmax(x, valid, out):
    row_groups = x.shape[0] // 32
    width_tiles = x.shape[1] // 32
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, width_tiles), block_count=2)
    valid_dfb = ttl.make_dataflow_buffer_like(valid, shape=(1, width_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, width_tiles), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(row_groups):
            with x_dfb.wait() as x_blk, valid_dfb.wait() as valid_blk, out_dfb.reserve() as out_blk:
                ones = ttl.block.fill(1.0, shape=(1, width_tiles))
                sentinel = ttl.block.fill(-128.0, shape=(1, width_tiles))
                z = x_blk * valid_blk + sentinel * (ones - valid_blk)
                row_max = ttl.math.reduce_max(z, dims=[1], shape=(1, 1))
                shifted = z - ttl.block.broadcast(row_max, dims=[1], shape=(1, width_tiles))
                weights = ttl.exp(shifted) * valid_blk
                denominator = ttl.math.reduce_sum(weights, dims=[1], shape=(1, 1))
                nonempty = ttl.math.reduce_max(valid_blk, dims=[1], shape=(1, 1))
                row_ones = ttl.block.fill(1.0, shape=(1, 1))
                safe_denominator = denominator + (row_ones - nonempty)
                inverse = ttl.recip(safe_denominator)
                out_blk.store(weights * ttl.block.broadcast(inverse, dims=[1], shape=(1, width_tiles)))

    @ttl.datamovement()
    def reader():
        for row in range(row_groups):
            with x_dfb.reserve() as x_blk, valid_dfb.reserve() as valid_blk:
                tx_x = ttl.copy(x[row:row + 1, 0:width_tiles], x_blk)
                tx_valid = ttl.copy(valid[row:row + 1, 0:width_tiles], valid_blk)
                tx_x.wait()
                tx_valid.wait()

    @ttl.datamovement()
    def writer():
        for row in range(row_groups):
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[row:row + 1, 0:width_tiles]).wait()
