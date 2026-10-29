# TT-Lang exercise source audit

Official repository: https://github.com/tenstorrent/tt-lang

Revision read: `a47484458bd964a6b92c9e4698aeb922585f08cc` (main resolved during this task).
Fetched source and blob hashes are in `ttlang-exercises/provenance.json`.
The two local teaching adaptations are `ttlang-exercises/vector_add_baseline.py`
and `ttlang-exercises/ragged_baselines.py`. Python syntax has been parsed. These
adaptations have **not** been compiled, simulated, emulated or hardware-tested.
Separately, the arithmetic-mask softmax formula was evaluated in NumPy FP32 for
64 rows at widths 32,96,256,544 including zero/singleton/boundary lengths: finite
outputs, zero invalid lanes, empty-zero semantics, nonempty normalization and
agreement with per-row valid-only stable softmax all passed. This checks the
mathematical formula, not compiler support or accelerator precision.

## Removing TT-NN from teaching while retaining truthful runtime boundaries

The student kernels can use only `import ttl`, with unannotated `a, b, out`
parameters. The course harness supplies preallocated device tensors and owns
device selection, memory allocation, transfer, warmup, timing and result checks.
This removes TT-NN baseline operations and TT-NN API instruction from the course.

It does not eliminate TT-NN from upstream TT-Lang's runtime: the official spec
says runtime operation arguments are TT-NN tensors only, and `ttl.operation`
functions return `None`, writing an explicit output tensor. No exported
`ttl.device_context` API or direct-Torch host invocation was found in this pinned
source. Do not invent either, or describe these operation definitions as plain
Torch-callable functions. All non-tensor configuration is compile-time, captured
from an enclosing factory/module scope, not an extra scalar runtime argument.

Sources:
- [Operation and runtime/compile-time argument specification](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/specs/TTLangSpecification.md#operation-function)
- [Exported TTL API](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/__init__.py)

The existing course OJ's `solve(a,b)` TT-NN contract does not establish support for
custom TT-Lang operations, compilation, or these new masks/outputs. Label the
TT-Lang contract a proposed course-harness interface until integrated. Do not
link a TT-NN-only problem as if the new TT-Lang kernel can be submitted unchanged.

## APIs that are actually implemented

- `@ttl.operation(grid=(1,1))`: one logical core for baseline. `grid="full"` plus
  `ttl.grid_size(dims=2)` and `ttl.node(dims=2)` allows distributing work. Coordinates
  are `(x,y)` / `(column,row)`. The upstream step-4 elementwise tutorial guards
  each trailing block in **all three kernels** so producer/consumer counts match.
- `ttl.make_dataflow_buffer_like(tensor,shape=(...),block_count=2)`: for tiled
  tensors, shape units are tiles, not scalar elements. One BF16 32×32 tile is
  2,048 bytes. `shape=(1,8)` is 32 scalar rows by 256 scalar columns.
- `with dfb.reserve() as blk:` reserves producer capacity, pushes on exit.
  `with dfb.wait() as blk:` waits for consumer data, pops on exit. Use explicit
  contexts; do not rely on simulator's extra implicit releases.
- `ttl.copy(src,dst)` is asynchronous. Its transfer `.wait()` must complete before
  reusing a source or publishing a destination. Multiple named in-flight handles
  are supported in explicit `@ttl.datamovement` kernels. The unified-body splitter
  currently rejects a saved handle waited later; there use `.wait()` on the call.
- `out_blk.store(a_blk+b_blk)` materializes a block expression into its reserved
  output. Source and destination shape/rank must agree exactly.
- `ttl.math.reduce_sum(x,dims=[1],shape=(1,1))` reduces **columns** of a `(1,Wt)`
  tiled block: 32 separate row sums, placed in scalar column 0 of the output tile.
  `ttl.math.reduce_max` has the same dimensional convention. `dims=[0,1]` instead
  reduces the entire block to a scalar, not row softmax.
- `ttl.block.broadcast(row_result,dims=[1],shape=(1,Wt))` expands those 32 row
  results across all scalar/tile columns. Use the explicit shape for reductions:
  omission is deprecated in compiler code and unsupported in Python simulator.
- `ttl.exp(x)` defaults to `approx=False` with clamping enabled;
  `ttl.recip(x)` computes reciprocal. `ttl.math.max(a,b)` is elementwise maximum.
- `ttl.block.fill(value,shape=(...),dtype=...)` defaults to BF16. If switching a
  teaching kernel to FP32, explicitly switch fills and relevant buffers/output.
  The implementation accepts Torch dtypes for this compile-time dtype selector.
- `ttl.block.where`, `ttl.block.mask`, and `ttl.block.mask_posinf` appear in the
  specification but are **not implemented by the compiler**. `ttl_block.py`
  states this explicitly and does not export them. Do not use them in a purported
  hardware baseline. A numeric 0/1 validity mask and finite-input arithmetic
  avoids this unsupported surface.

Sources:
- [Full-grid elementwise example](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/elementwise-tutorial/step_4_multinode_grid_full.py)
- [Compiler-supported block namespace](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_block.py)
- [Reduction, fill, typecast and exp implementation](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/operators.py)
- [Reduction hardware tests](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/python/test_reduce.py)
- [Simulator reduction placement](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/sim/math.py)

Beware two misleading upstream descriptions: the spec's reduction table swaps
rows/columns in parts of its prose, while implementation/tests use ordinary axis
semantics; `softmax_compiler_allocated_dfb.py` calls itself row-wise, but actually
uses `dims=[0,1]` and its golden result flattens each 32×32 tile. Teach from the
executed formula, not those headings.

## Exercise 1: large vector add

Contract: `c[i]=a[i]+b[i]` for `0<=i<N`, finite BF16 inputs. Harness pads to
`Npad=1024*ceil(N/1024)` and reshapes to `[32*ceil(N/1024),32]`. The last physical
tile exists; only first N unpacked values are logical outputs. No scalar-level
out-of-bounds tensor slice is needed. Student baseline performs one tile per
stream iteration on one core; reader, compute, writer all process `Npad/1024`
entries. Basic BF16 explicit DFB storage is `3*2*2048=12,288` bytes/core.

Proposed tests (no measured performance claims):

| N | Intended observation |
|---:|---|
| 1,024 | Launch/synchronization overhead dominates little useful work. |
| 65,536 | Enough work to compare core count and transfer granularity. |
| 1,048,576 | Streaming bandwidth and double buffering. |
| 16,777,216 | Sustained throughput, memory capacity and batching. |
| 1,048,577 | One extra physical tile; avoid dropping the tail or counting padding as useful work. |

Guided changes: blocks of 1,2,4,8 tiles; one core to full grid; identical guarded
partition in all three threads; inspect time waiting on input/output buffers;
report useful bandwidth `3*N*2/time` for BF16 and physical bytes separately.
Larger blocks reduce transfer/synchronization count but reserve more L1. A pure
one-op vector add has essentially no data reuse, so compute optimization alone
will not eliminate the input/output traffic.

Source: [official full-grid add](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/eltwise_add.py).

## Exercise 2: ragged sum

Contract: padded BF16 `x[Rpad,Wpad]`, numeric BF16 `valid[Rpad,Wpad]` with 1 for
`j<length[r]` and 0 elsewhere; finite inputs. `s[r]=sum(x[r,j],j<length[r])`;
empty rows return zero. Harness must supply mask and true int lengths without
encoding lengths in BF16. Kernel consumes masks rather than branching per row.
The baseline output is tiled `[Rpad,32]`, logical values in column zero. Compare
only logical rows/column zero, not unspecified padding lanes of the reduction.

For a 32-row group and Wpad<=256:

```python
with x_dfb.wait() as x_blk, valid_dfb.wait() as v_blk, out_dfb.reserve() as y_blk:
    y_blk.store(ttl.math.reduce_sum(x_blk * v_blk, dims=[1], shape=(1, 1)))
```

Start shapes: 32 rows with lengths repeated from `[0,1,31,32,33,63,64,65]`, Wpad96;
1024 rows of width32; 1024 rows of width256. Then equal-useful-work profiles below
expose padding efficiency versus scheduling. A dense fixed-width masked kernel
does identical physical work regardless of lengths; the skew benefit appears
only when the implementation buckets lengths/skips all-invalid blocks/changes
representation. Report that explicitly.

Precision: upstream reduction tests cover BF16 and FP32 separately because LLK
paths differ; BF16 truncation grows across many tiles. Compute CPU golden in FP32
from the **quantized input values**. Report maximum absolute error and a relative
metric with a floor for cancellation; do not rely on PCC alone or demand exact
agreement across different reduction orders. Wider DST does not undo a BF16
intermediate stored between chunks.

## Exercise 3: ragged softmax

Contract uses the same mask with finite logits in [-80,80]. Empty rows return all
zeros; invalid lanes are exactly zero; valid lanes of nonempty rows sum to about
one under the adopted precision tolerance. Include rows with identical logits,
one valid item, large offsets, one dominant logit, and all-invalid rows.

Implemented-API formula (full kernel is in the companion source):

```python
ones = ttl.block.fill(1.0, shape=(1, WT))
sentinel = ttl.block.fill(-128.0, shape=(1, WT))
z = x_blk * v_blk + sentinel * (ones - v_blk)
m = ttl.math.reduce_max(z, dims=[1], shape=(1, 1))
e = ttl.exp(z - ttl.block.broadcast(m, dims=[1], shape=(1, WT))) * v_blk
d = ttl.math.reduce_sum(e, dims=[1], shape=(1, 1))
nonempty = ttl.math.reduce_max(v_blk, dims=[1], shape=(1, 1))
safe_d = d + (ttl.block.fill(1.0, shape=(1, 1)) - nonempty)
y_blk.store(e * ttl.block.broadcast(ttl.recip(safe_d), dims=[1], shape=(1, WT)))
```

`-128` is below every valid input by the stated contract. Nonempty rows have a
valid maximum; empty rows produce z=-128, exp(0)=1 then e=0 due to valid=0, and
safe_d=1. This avoids `-inf-(-inf)` and division by zero. Arithmetic masking is
not safe for arbitrary NaN/Inf input because `0*NaN` remains NaN. Do not quietly
widen the contract to arbitrary floating-point values.

The compiler may allocate intermediate DFBs for reduce/broadcast/exp boundaries;
the visible x/mask/out buffers are not the complete L1 footprint. The official
softmax test checks generated `reduce_tile`, pack/push/wait, broadcast,
`exp_tile`, `recip_tile`, `mul_binary_tile`. Students should find those boundaries
in their own artifact, count extra materializations, and compare with explicit
staged DFBs only after a correct simple version.

Source: [compiler-allocated softmax DFB test](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/python/softmax_compiler_allocated_dfb.py).

## Long rows and workload families

| Family | Logical lengths | Useful elements | Question |
|---|---|---:|---|
| Short/many | 4,096×32 and 4,096×96 | 524,288 | Launch/amortization, row-group parallelism, grouping by width. |
| Balanced | 1,024×542 | 555,008 | Moderate width, tile-tail overhead; Wpad544. |
| Skew | 960×32 and 64×8,192 | 555,008 | Same useful work, severe dense padding and uneven per-row work. |
| Long/few | 32×16,384 | 524,288 | Limited row parallelism, chunked reduction/replay, L1 capacity. |

These are advanced chunked/bucketed workloads, not all valid sizes for the small
full-row baseline. At Wpad8192, three BF16 double-buffered full-row streams for
32 scalar rows require `3*2*32*8192*2 = 3,145,728` bytes **before** intermediates.
Do not present them as fitting a core just because the tensors fit DRAM.

Ragged sum extension: use a fixed tile-column chunk width, reduce each chunk to
32 row values, and accumulate partials with an explicit lifetime. Zero masked
padding in the last chunk; compare BF16-stored vs FP32-stored state. For full-grid
row assignment, compare contiguous equal-row groups with length buckets. Splitting
one long row over cores adds a second reduction stage; no free speedup claim.

Softmax extension: a clean three-pass algorithm (max, shifted sum, normalized
write) streams fixed-size chunks. A two-pass variant computes online `(m,l)` in
pass1 and rereads original logits to normalize in pass2:

`m_new=max(m_old,chunk_max)`;
`l_new=exp(m_old-m_new)*l_old + sum(valid*exp(chunk-m_new))`.

Use finite sentinel initialization, zero invalid weights, explicit empty-row
handling, and state buffers. Each chunk's new sum depends on its new max.
Retaining all exponentials can save a read but costs L1/DRAM and changes traffic;
ask students to quantify this rather than calling the variant universally faster.
The official online-softmax benchmark implements dependent m/l/o state for
attention, making it a useful advanced reference rather than a copy-paste
ragged-softmax answer.

Source: [online softmax accumulator benchmark](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/benchmarks/softmax/online_softmax_accumulators.py).

## Simulator and measurement boundaries

The Python `tt-lang-sim` backend runs logic with Torch-backed tensors without
hardware/compiler. It supports more language features than the compiler and
normally promotes BF16/narrow floats to FP32. `--no-float32-promotion` tests the
declared dtype more closely, but neither mode proves hardware numerics or speed.
The simulator still uses declared dtypes for L1 accounting and warns, rather than
aborts, when exceeding L1. Its convenience auto-release of blocks can hide
missing push/pop; use explicit contexts in all teaching code. Simulator traces
describe dataflow, not measured accelerator performance.

The source-tree emule backend runs compiler + runtime + emulator; it is a
separate path from Python simulation. No backend was executed in this audit.
Use real pinned-device runs for warm kernel timing, compile times, assembly and
performance claims. Keep simulator/emulator/hardware provenance separate.

Source: [official simulator guide](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/simulator.md).
