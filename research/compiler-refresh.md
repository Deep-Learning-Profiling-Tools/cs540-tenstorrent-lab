# Compiler chapter refresh: verified facts and tools

Audit revision: `a47484458bd964a6b92c9e4698aeb922585f08cc`.
This is source inspection, not compiler execution. Official source copies are in
`compiler-refresh-sources/`; the pinned `ttl_api.py` and compiler options are also
already available under the earlier research directories.

## Correction required in the current website

`TTLANG_FINAL_MLIR` is **not a guaranteed TTKernel checkpoint before EmitC**.
The Python pipeline explicitly appends `ttl-lower-signpost-to-emitc`,
`convert-ttkernel-to-emitc`, and `symbol-dce`, runs that pipeline, and only then
writes `TTLANG_FINAL_MLIR`. Name the file `final.mlir`, describe it as post-pipeline
IR, and have students inspect its actual dialects. The existing audit and website
statement claiming pre-EmitC TTKernel is incorrect at this pin.

Evidence: [Python pipeline tail and final dump](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3655-L3721).
For a clean standalone TTKernel experiment, run the registered pipeline with
`lower-to-emitc=false` against saved initial IR and **label it a separate run**
unless all original options and passes were matched. The actual Python pipeline
is printed by `TTLANG_VERBOSE_PASSES=1`; use its before/after-pass log when a
checkpoint from the same compilation is required.

The C++ registered pipeline and Python path share the core transformation order,
but their tails are not literally identical: Python additionally performs print
and signpost lowering, `ttkernel-lower-scalar-fp-types`, and `symbol-dce`; the C++
pipeline's optional EmitC tail is convert → canonicalize → form expressions.
Avoid saying a default standalone command reproduces every Python option.

## Suggested chapter separation

**Chapter 02: How a TT-Lang program is compiled.** Build the conceptual path and
show what each representation expresses. Use one small matmul tile example and
its shape/indexing facts. Introduce SSA values, blocks/regions, dialect names,
attributes, tensor tile types, loop recurrence, and the distinction between
logical DFB identity and physical allocation. End with one source-backed
compiler fixture to read before asking students to run tools.

**Chapter 04: Compiler transformations and inspection lab.** Reuse that example;
identify the observable change made by representative passes, capture artifacts,
run one isolated pass, compare one option, then connect generated C++ to actual
device binary evidence. Keep a compact tools catalogue here. Avoid duplicating a
large unexplained pipeline list in both chapters.

## Stage map and what students should read

| Representation | What it represents | Concrete thing to inspect |
|---|---|---|
| Python operation | Explicit compute/data movement kernels, grid, DFB shapes, block expressions and source control flow | Which statements execute on which core/thread; which values are compile-time metadata versus runtime tensors |
| Initial TTL MLIR | Typed block arithmetic, DFB transactions/copies, structured loops and SSA recurrence | `ttl.bind_cb`, `ttl.cb_wait/reserve`, `ttl.attach_cb`, `ttl.matmul`, `scf.for`/`iter_args`, source locations |
| TTL compute/tile IR | Iteration space and operation recipes at tile granularity | `ttl.compute`, indexing maps, iterator kinds, tile operations and stores |
| Allocated/scheduled TTL | Concrete DFB identities/storage, DST placement, subblocks and loop order | Assigned DFBs, `dst` indices, inserted copies/materializations, synchronization scopes |
| TTKernel | Target-facing buffer, NoC, register, pack and math API operations | Which API initialization is inserted, where math/pack synchronization occurs, arguments and loop bounds |
| EmitC | A representation suitable for C++ emission | Calls/types and control constructs replacing target dialect operations |
| Generated C++ | Device source passed to the later device compiler/runtime | Separate reader, writer and compute functions; init, copy, wait, pack, pointer/index arithmetic |
| Device ELF/disassembly | Concrete code produced by the recorded device toolchain | Actual control flow, instruction issue, registers/addresses, kernel identity and target |

`tensor<2x4x!ttcore.tile<32x32,bf16>>` contains eight 32×32 BF16 tiles, i.e.
64×128 scalar values. For tile-grid matmul `(2,4)@(4,3)->(2,3)`, use loop coordinates
`(m,n,k)`, maps `(m,k)`, `(k,n)`, `(m,n)` and iterator kinds parallel, parallel,
reduction. The output block contains six tiles. Do not relabel this whole-block
fixture as the successive output of a one-output-tile Python baseline.

Sources:
- [Authentic matmul TTL-to-compute fixture](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/ttlang/Conversion/TTLToCompute/matmul_to_compute.mlir)
- [Authentic matmul TTKernel lowering fixture](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/ttlang/Conversion/TTLToTTKernel/matmul_block_lowering.mlir)
- [C++ translation binding](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttlang/PassesModule.cpp)

## Precise core pass ordering at this revision

Authoritative registered pipeline:
[TTLPipelines.cpp](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp).
Python sequence:
[ttl_api.py](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3557-L3673).

1. `ttl-form-accumulation-scopes` → `ttl-lower-accumulation-scopes`
   (tensor recurrence, selected `auto`/`dst`/`l1-pack` strategy).
2. `ttl-materialize-loop-state` → `ttl-insert-copy-wait`.
3. `ttl-insert-cb-sync` → `ttl-coalesce-dfb-acquires`.
4. `ttl-insert-accumulation-scopes{kind=dfb}` →
   `ttl-lower-accumulation-scopes{kind=dfb}`.
5. `ttl-create-producer-compute` → `ttl-insert-intermediate-dfbs` →
   `convert-ttl-to-compute` → `ttl-insert-cb-sync`.
6. `ttl-verify-pipenet-guards` → `ttl-verify-pipenet-schedule` →
   `ttl-form-pipe-transports` → `ttl-coalesce-dfb-acquires`.
7. `ttl-finalize-dfb-indices` → `ttl-set-compute-kernel-config` → `ttl-assign-dst`.
8. `ttl-subblock-compute-for-dst` (if maximize DST) → `ttl-lower-to-loops` →
   `ttl-schedule-operations` (if maximize DST) → `ttl-annotate-cb-associations`.
9. `ttl-verify-dfb-spsc` → `ttl-verify-dfb-lifecycle` → `ttl-erase-pipenet-scopes`
   → `ttl-validate-cb-budget`.
10. `convert-ttl-to-ttkernel` → `ttkernel-insert-inits` →
    `ttkernel-insert-l1-accumulation` → `ttkernel-combine-pack-tiles` (if enabled).
11. Canonicalization/CSE, then common record cleanup or optional core specialization.
12. Optional EmitC conversion in registered C++ pipeline; Python always appends
    the aforementioned EmitC/signpost/symbol-DCE tail at this pin.

Record cleanup is: static PipeNet receive batching → static record-loop unrolling
→ lower affine → canonicalize → CSE → `ttkernel-cleanup` →
`ttkernel-finalize-tensor-runtime-args` → canonicalize. Specialization wraps that
with core cloning/constant folding first, then specialized DFB reconfiguration
and surviving DFB-use annotation afterward.

The didactic article [LOWERING_MULTITILE.md](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/LOWERING_MULTITILE.md)
is useful for illustrated IR, but it still lists a separate
`ttl-insert-tile-regs-sync` stage absent from the current registered pipeline.
Do not copy that stage list as the exact current pipeline.

## Representative transformations: teach the cause and visible effect

1. **Recurrence storage:** a tensor carried through `scf.for` must have a legal
   place to live between iterations. Scope formation identifies the recurrence;
   strategy lowering selects DST/L1 mechanisms; remaining loop state gets DFBs.
   Compare `dst` with `l1-pack` and inspect both generated storage and error.
   A register-resident FP32-looking recurrence is not necessarily full FP32:
   the documented DST feedback path truncates through TF32 precision.
2. **Compute formation and fusion:** supported producers can become tile recipes
   in one `ttl.compute`, avoiding intermediate DFB traffic. Fusion must preserve
   DFB input availability, output transactions, result dominance, and observable
   instrumentation order. If an expression outlives its input's release or needs
   a DFB-backed operand, intermediate materialization is required.
3. **Physical DFB allocation:** logical DFBs can share physical storage only with
   compatible nonoverlapping lifetimes. Pipeline order checks PipeNet domains
   and schedules before physical identities are merged, then validates final
   ownership/lifecycle/capacity afterward.
4. **Configuration then DST allocation:** tile operation alternatives and the
   target jointly constrain DST width/unpack modes. Strategy is recorded before
   DST assignment; the allocator uses that selection rather than rediscovering
   it after copy insertion changes the operands.
5. **Subblocking and scheduling:** a large tile-grid expression is partitioned
   into register-sized work. This changes synchronization granularity and may
   cause accumulator materialization. Grouping more operations in a DST section
   can reduce handshake overhead; it is constrained by live operands and outputs.
6. **Pack combining:** adjacent compatible `pack_tile` operations with contiguous
   DST and DFB indices can become `pack_tile_block`. Inspect both generated calls
   and the number of synchronization scopes before interpreting a speed result.
7. **Optional core specialization:** core-coordinate-dependent branches/loops
   become constants in per-coordinate functions; cleanup removes unreachable
   paths/table entries. Identical generated code plus runtime metadata can share
   a descriptor. Specialization is off by default.

Design references:
- [Compute creation/fusion and lifetime legality](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/ComputeOpCreation.md)
- [Kernel configuration and target restrictions](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/ComputeKernelConfiguration.md)
- [DST allocation](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/DST_Allocation.md)
- [DST utilization](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/DST_Utilization.md)
- [Static execution cardinality](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/StaticExecutionAnalysis.md)
- [Value origins through control flow](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/development/ValueOriginAnalysis.md)

## Compact inspection-tool catalogue

| Tool/API | Input → evidence | Example / limitation |
|---|---|---|
| Python runner + dump environment | Actual invoked operation → initial/final IR, pass logs, generated-source paths | Must call the operation, not merely import a kernel-only file |
| `ttlang-opt` | Saved MLIR → one pass or registered pipeline output | `--help` lists supported flags; `-p` selects a pipeline |
| `FileCheck` | Generated text + expected patterns → regression assertions | `// CHECK` comments are patterns, not captured output |
| `llvm-lit` | Upstream test file → executes its RUN lines | Useful when reproducing the fixture in a configured checkout |
| `ttlang-translate --ttkernel-to-cpp` | EmitC/target-ready module → generated C++ | Verified exact flag in translator registration and translation tests |
| `ttl-dump-cb-flow-graph` pass | TTL module → DFB producer/consumer JSON | Gives buffer relationships, not measured transfer time |
| Device debug `print` | Instrumented device execution → scalar/DFB/tile contents | Select a few cores/threads; it perturbs timing |
| `TTLANG_AUTO_PROFILE` | Instrumented hardware run → per-source-line cycles | Keep separate from targeted signpost mode; trace buffer is limited |
| `ttl.signpost` + `TTLANG_SIGNPOST_PROFILE` | Named source regions → per-thread cycle counts | Nested/concurrent regions cannot simply be summed into wall time |
| `TTLANG_PERF_DUMP` | Hardware profiler logs → NoC traffic and per-kernel summary | Requires device profiler and NoC events |
| `TTLANG_PERF_SERV` | Profiler data → Perfetto timeline | Follow printed connection instructions for remote machines |
| `tt-lang-sim` + `tt-lang-sim-stats` | Logical execution → dataflow trace/counts | Python simulation is not accelerator timing or default BF16 numerics |
| Matching SFPI `riscv-tt-elf-objdump` | Recorded kernel ELF → disassembly | Requires artifact association and matching target/toolchain |

### Capture one actual compilation

```bash
mkdir -p artifacts
TTLANG_INITIAL_MLIR=artifacts/initial.mlir \
TTLANG_FINAL_MLIR=artifacts/final.mlir \
TTLANG_VERBOSE_PASSES=1 \
TTLANG_DEBUG_LOCATIONS=1 \
TTLANG_VERBOSE_ERRORS=1 \
TTLANG_VERBOSE_KERNELS=1 \
python "${COURSE_RUNNER:?Set the actual host runner}" \
  > artifacts/compiler.log 2>&1
```

`VERBOSE_PASSES` prints before all passes, after all passes, and after failure in
the actual Python implementation. `DEBUG_LOCATIONS` adds printed source location
information. Generated C++ paths normally use
`/tmp/$USER/ttlang_kernel_<name>_<content-hash>.cpp` (pytest workers can add a
subdirectory); copy only the printed files associated with this run. Reuse of a
cached compiled operation can avoid the dump path, so capture a real compilation
in a fresh process and verify that files were produced.

`TTLANG_COMPILE_ONLY=1` skips operation execution. It does not remove the device
setup requirements of the host runner or guarantee later JIT-generated ELF
artifacts; skip numerical checks in a compile-only harness.

### Standalone pass and translation experiments

```bash
# Reproduce the downloaded fixture's isolated conversion.
ttlang-opt matmul_to_compute.mlir \
  --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' \
  --split-input-file -o compute-tests.mlir
FileCheck matmul_to_compute.mlir --input-file=compute-tests.mlir

# Separate full-pipeline experiment: record options and target attributes.
ttlang-opt artifacts/initial.mlir \
  -p 'ttl-to-ttkernel-pipeline{lower-to-emitc=false}' \
  -o artifacts/standalone-ttkernel.mlir
ttlang-opt artifacts/initial.mlir \
  -p 'ttl-to-ttkernel-pipeline{lower-to-emitc=true}' \
  -o artifacts/standalone-emitc.mlir
ttlang-translate --ttkernel-to-cpp \
  artifacts/standalone-emitc.mlir -o artifacts/standalone-kernels.cpp

# DFB relationship analysis (use an appropriate pre-conversion checkpoint).
ttlang-opt artifacts/initial.mlir \
  -p 'ttl-dump-cb-flow-graph{output="artifacts/cb-graph.json"}' \
  -o artifacts/graph-input.mlir
```

Translation flag evidence:
[registration](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/lib/Target/TTKernel/TTKernelToCppRegistration.cpp),
[official translate test commands](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/ttlang/Translate/TTLToCpp/compute_fused_chain_to_cpp.mlir).
The test uses `--allow-unregistered-dialect` for its fixture pipeline; do not add
that as a blanket way to suppress invalid student IR. The command above assumes
the normal registered full-pipeline output.

### One-variable compiler experiments

```bash
python "$COURSE_RUNNER" --ttl-help
python "$COURSE_RUNNER" --no-ttl-maximize-dst
python "$COURSE_RUNNER" --ttl-accumulation-strategy dst
python "$COURSE_RUNNER" --ttl-accumulation-strategy l1-pack
python "$COURSE_RUNNER" --no-ttl-combine-pack-tiles
```

Also available: `TTLANG_COMPILER_OPTIONS="..."` and decorator `options="..."`.
Documented priority is defaults < decorator < environment < CLI. Call-time
`options=` can replace decorator settings for one invocation. Keep one deliberate
setting source per experiment. Do not suggest `--no-ttl-block-matmul`: disabling
that option is explicitly unsupported at this pin.

### Device inspection and profiling

```bash
# Small-input debug run, not a timing run.
TT_METAL_DPRINT_CORES=0,0 python "$COURSE_RUNNER"

# Hardware NoC/traffic summary (TT_METAL_HOME set by course environment).
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
TTLANG_PERF_DUMP=1 python "$COURSE_RUNNER"

# Analyze saved logs, independently of a new run.
python -m ttl._src.perf_summary --path /path/to/profiler/.logs/ --json
python -m ttl._src.perf_trace_server --path /path/to/profiler/.logs/
```

In kernel code: `print(tile, thread="pack")` while the block is live;
`print(inp_dfb)` for metadata; `print(_dump_dst_registers=True,label="after exp")`
for BF16 DST debugging. The documentation says tile printing is unsupported on
the math thread despite showing `thread="math"` in a generic syntax example;
teach the supported `pack` form. Tensor page prints belong on data movement
threads. Remove debug output for performance comparison.

For named measurements use `with ttl.signpost("compute"):` and enable
`TTLANG_SIGNPOST_PROFILE=1`; for automatic source-line attribution use
`TTLANG_AUTO_PROFILE=1`. They must not be enabled together. Both require the
device profiler and mid-run dumps. The pinned guide describes a 125-signpost
per-node limit; reduce the measured iteration count and inspect overflow warnings.
`TTLANG_PERF_SERV=1` can accompany either mode for Perfetto.

### Simulator trace inspection

```bash
tt-lang-sim "$COURSE_RUNNER" --trace artifacts/sim.jsonl
tt-lang-sim-stats artifacts/sim.jsonl
tt-lang-sim --no-float32-promotion "$COURSE_RUNNER"
```

The Python simulator generally promotes narrow floats to FP32, supports some
features absent from the compiler, and provides convenience release behavior
that should not replace explicit queue ownership. No hardware timings or
assembly should be inferred from its results. Compiler-backed emulation is a
separate source-tree backend (`./bin/tt-lang-sim --backend=emule ...`).

### Primary tool references

- [Compiler options, environment variables and pass reference](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/reference/compiler-options.md)
- [Device print debugging](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/reference/print-debugging.md)
- [Hardware performance tools](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/reference/performance-tools.md)
- [Simulator and trace statistics](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/simulator.md)
- [Build/test tools](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/tools.md)
- [Implementation guide index](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/implementation.md)
