> Correction from the 2026-10-16 source refresh: the original FINAL_MLIR placement claim below was wrong. The Python pipeline writes it after EmitC conversion. See compiler-refresh.md and the current course chapter for corrected commands.

# TT-Lang source audit — 2026-10-16

All TT-Lang sources below are pinned to **a47484458bd964a6b92c9e4698aeb922585f08cc**. They are upstream examples or compiler tests, not output from a course-device run. This audit did not execute Tenstorrent kernels or measure performance.

## Teaching scope and stack

Teach TT-Lang Python operations: explicit data movement, tiled block math, DFB ownership, and concurrency. Public examples still use TT-NN tensor containers for host allocation, upload, and download. That plumbing can live in a collapsed host helper; it does not require teaching TT-NN operation APIs.

At this pin, a search of the public repository did not find a Python `device_context` API. Do not invent one. Official examples use `ttnn.open_device`, `ttnn.from_torch`, `ttnn.to_torch`, and `ttnn.close_device`.

The compiler path is **TT-Lang Python AST → TTL MLIR → TTL compute/tile operations → TTKernel MLIR → EmitC → generated Metalium C++ → SFPI/RISC-V device code**. StableHLO, TTIR, and TTNN dialects belong to a different frontend path and should be removed from this tutorial.

Primary implementation evidence: [Python compiler pipeline](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3481), [EmitC translation binding](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttlang/PassesModule.cpp#L47).

## Matmul progression

1. [Single-node pure matmul](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/single_node_matmul.py): manageable 128×256 @ 256×64 host example. `acc = ttl.block.fill(0, shape=out_blk.shape)`; then `acc += a_blk @ b_blk` across K; finally store. This is the simplest full example without bias/ReLU. It imports `utils.correctness.assert_pcc`; copying it alone requires keeping or replacing that host helper.
2. [Step 1](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/matmul-tutorial/step_1_single_node_single_tile_block.py): one node, one tile per block; complete reader/compute/writer; explicit accumulator DFB ping-pong. Computes `relu(A@B+C)`. Upstream dimensions are 8192³ and should be reduced for introductory course examples.
3. [Step 2](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/matmul-tutorial/step_2_single_node_multitile_block.py): M/K/N block granularities all 4; A DFB=(Mblock,Kblock), B=(Kblock,Nblock), accumulator/output=(Mblock,Nblock). Block shapes count tiles, not scalar elements.
4. [Step 3](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/matmul-tutorial/step_3_multinode.py): grid=(4,4); each node owns an output rectangle; K remains local/full. All dimensions must divide the assumed tiling and partitioning.
5. [Step 4](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/matmul-tutorial/step_4_multinode_grid_full.py): grid="full"; `grid_n,grid_m=ttl.grid_size(dims=2)`; `node_n,node_m=ttl.node(dims=2)`; ceil division distributes complete blocks; all three threads guard the same block conditions.
6. Reuse and multicast extension: [1D multicast TT-Lang](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/metal_examples/1d_mcast_matmul/ttlang/1d_tt_lang.py), [2D multicast TT-Lang](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/examples/metal_examples/2d_mcast_matmul/ttlang/2d_mcast_matmul.py), with paired Metalium implementations in adjacent directories. These connect directly to Metalium labs.

**Step 4 handles a number of complete blocks that does not divide the grid. It does not handle arbitrary element tails:** block counts still use floor division by 32 and block granularity. Off-tile shapes require an explicit padded input/output contract or additional tail work.

## Exact basic API and ownership

```python
@ttl.operation(grid=(1, 1))
def operation(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def reader():
        with a_dfb.reserve() as a_blk:
            ttl.copy(a[0, 0], a_blk).wait()

    @ttl.compute()
    def compute():
        with a_dfb.wait() as a_blk:
            # block arithmetic and output store
            ...
```

- An operation returns None and writes its output tensor argument.
- DFB `shape=(1,1)` is one 32×32 tile in these examples; BF16 payload is 2048 bytes. `block_count=2` allocates two blocks per participating node, permitting producer/consumer overlap.
- `reserve()` waits for a free block. Scope exit publishes it (`push`).
- `wait()` waits for a filled block. Scope exit releases it (`pop`).
- `ttl.copy` starts a transfer; `.wait()` completes it before the block is published/released.
- Compute, reader, and writer execute concurrently; function declaration order is not their execution schedule.
- DFB queue order and equal production/consumption counts connect the threads.
- Explicit accumulator DFB ping-pong needs two blocks: compute holds the old block while reserving and publishing the new block. A one-slot queue can deadlock.
- A mathematical accumulator is not automatically evidence that every partial value stays in DST registers. Compare generated IR/code and `--ttl-accumulation-strategy`.
- For fixed grid, coordinates are (N,M) in the tutorial, so output row ownership uses the second coordinate.
- The language spec also describes unified-body operations. For this course, the verified explicit-thread examples best expose Metalium concepts.

Source: [language spec](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/specs/TTLangSpecification.md), and the exact downloaded examples.

## Genuine IR reading material

- `matmul_to_compute.mlir`: upstream input fixture. `ttl.matmul` on `tensor<2x4x!ttcore.tile<32x32,bf16>>` @ `tensor<4x3x!ttcore.tile<32x32,bf16>>` corresponds to 64×128 @128×96 scalars. FileCheck comments expect `ttl.compute`, affine maps M,K / K,N / M,N, parallel M/N and reduction K, and `ttl.tile_matmul_block`.
- `matmul_block_lowering.mlir`: upstream input and expected TTKernel patterns. Shows register acquire/commit/wait/release, `ttkernel.matmul_block`, and pack operations. Comments are expected patterns, not captured output.
- `simple_matmul_test.py`: real one-tile frontend and C++ checks; expects `ttl.matmul`, `matmul_block_init`, `matmul_block`, and `pack_tile`.

These fixtures do not require pretending we captured a complete frontend-to-assembly run.

## Artifact capture commands

Documented flags and source-verified output behavior:

```bash
mkdir -p artifacts
TTLANG_INITIAL_MLIR=artifacts/initial.mlir \
TTLANG_FINAL_MLIR=artifacts/final-ttkernel.mlir \
TTLANG_VERBOSE_PASSES=1 \
TTLANG_DEBUG_LOCATIONS=1 \
TTLANG_VERBOSE_KERNELS=1 \
python matmul.py > artifacts/compiler.log 2>&1
```

This normal hardware command compiles and executes the student's host wrapper. Do not run the unmodified 8192³ single-core tutorial as an introductory experiment.

`TTLANG_INITIAL_MLIR` is written before the optimization/lowering pipeline. `TTLANG_FINAL_MLIR` is written after the TTL→TTKernel pipeline **before** EmitC conversion. Per-pass IR goes to the redirected log. Generated C++ and its real path are printed by default; `TTLANG_VERBOSE_KERNELS=0` suppresses that output. Files are written at `/tmp/$USER/ttlang_kernel_<name>_<content-hash>.cpp`, optionally under a pytest worker subdirectory. Copy the specific printed files into the report bundle.

To separately inspect EmitC from the saved initial IR in an installed compiler environment:

```bash
ttlang-opt artifacts/initial.mlir \
  -p 'ttl-to-ttkernel-pipeline{maximize-dst=true lower-to-emitc=true}' \
  -o artifacts/emitc.mlir
```

This standalone pipeline may differ from the original run's options. Match explicit options or label it a separate compiler experiment.

`TTLANG_COMPILE_ONLY=1` skips operation execution; it does not make a host wrapper device-free. The official examples still open the device/upload tensors, and their output assertions fail if skipped computation is mistaken for a completed run. A compile-only teaching harness must skip those result checks. It also does not guarantee a device ELF: TT-Metal JIT compilation is a later execution/runtime step.

Valid experiments include `--no-ttl-maximize-dst`, `--ttl-accumulation-strategy dst` vs `l1-pack`, and `--no-ttl-combine-pack-tiles`. The options table explicitly says disabling block matmul is not supported. `--ttl-help` prints available options.

Source: [compiler options](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/reference/compiler-options.md), [temporary C++ writer](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L1224), [final MLIR output](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3712), [execution guard](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3935).

## Assembly evidence

Do not claim TT-Lang offers a verified "dump assembly" switch: none was found. Use the TT-Metal generated ELF from the recorded runtime, then the **matching SFPI** disassembler. Current official LLK documentation names `riscv-tt-elf-objdump -d`; older packages may use other binary names.

Suggested workflow, with actual paths supplied by the instructor:

```bash
/path/to/matching/sfpi/compiler/bin/riscv-tt-elf-objdump -d \
  /path/to/recorded/kernel.elf > artifacts/kernel.disasm
```

This is a parameterized command, not a verified course-server path. Preserve the ELF, target, SFPI version, generated source, runtime options, and kernel association. TT-Metal caches binaries under `TT_METAL_CACHE`, but locating a file alone is insufficient to associate it with the matmul run.

Primary evidence: [TT-Metal JIT cache documentation](https://github.com/tenstorrent/tt-metal/blob/7c3dd90a675f0f00316786d01e4f60be42c3c9c0/tt_metal/tools/jit_compile_server/README.md), [official LLK disassembly workflow](https://github.com/tenstorrent/tt-metal/blob/7c3dd90a675f0f00316786d01e4f60be42c3c9c0/tt_metal/tt-llk/.claude/skills/perf-optimization-audit/references/validation-and-output.md). The latter is evidence only; its unrelated workflow instructions were not adopted.

The earlier Blackhole macro excerpt remains useful as an **independent authentic inline-assembly source example**, never as a matmul disassembly. `.ttinsn` issues Tensix instructions and must not be confused with completion or a single high-level matmul.

## Ragged exercises

At this pin the spec's support matrix lists `ttl.block.where`, `mask`, and `mask_posinf` as unsupported by the compiler. Do not publish baselines that rely on them. Host-created numeric masks plus multiplication and explicit finite sentinel arithmetic are safer design directions; retain finite-input assumptions and explicit empty-row handling. No proposed ragged TT-Lang baseline was hardware validated here.

Use `ttl.math.reduce_sum(x, dims=[-1])` or explicit `shape=(row_tiles,1)`, and `ttl.block.broadcast(..., dims=[-1], shape=x.shape)` to expand a row result. Verify the exact semantics through existing reduction tests. A reduction over both [0,1] reduces **every scalar in each block**, not a rowwise reduction.

`softmax_compiler_allocated_dfb.py` is genuine compiler-tested source but despite its introductory "row-wise" wording, its actual kernel and CPU reference normalize each complete 32×32 tile over all 1024 elements. Use it to study compiler-created intermediate DFBs; do not reuse its dimensions as a row-softmax answer.

## Attribution

All copied TT-Lang sources are licensed Apache-2.0. The directory includes upstream LICENSE and NOTICE. Preserve SPDX/copyright headers when adapting examples and mark modified files as adapted. Metadata with exact source URLs and Git blob hashes is in `ttlang/provenance.json`.
