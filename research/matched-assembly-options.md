# Matched TT-Lang → IR → C++ → assembly options

Research checked 2026-10-16. Stopped before installation, compilation, or runtime setup at the user's request to use placeholders for now. **No actual matched course artifact chain was produced.** These findings are for a later capture task.

## Strongest verified results

1. TT-Lang supports authentic device-free frontend → IR → generated C++ compilation using **host TTNN tensors** as metadata and `TTLANG_COMPILE_ONLY=1`. An upstream lit test exercises precisely this.
2. TT-Metal exposes an **experimental offline kernel compiler**, which can generate actual target ELFs without a live accelerator. It requires compatible runtime libraries, SFPI, firmware, and exact compile metadata.
3. `TT_METAL_KERNEL_MAP=1` automatically adds `-save-temps=obj -fdump-tree-all -fdump-rtl-all` at compile and link time in the pinned TT-Metal source. Together with compile-command logging, this is the useful actual-assembly capture path.
4. No ready-made, complete saved matmul chain (same Python source → full actual TTL/TTKernel → generated C++ → device assembly) was found in the inspected TT-Lang source tree. Its .mlir regression fixtures and Metalium C++ examples are independent test/example sources, not consecutive captured dumps.

Pins:
- TT-Lang: `a47484458bd964a6b92c9e4698aeb922585f08cc`.
- That TT-Lang checkout's **tt-metal submodule gitlink**: `0e9d200db976120c129ab0deb13aa3f6d972b723`. Use this compatibility pin, not an independently chosen latest tt-metal.
- SFPI README audit: `21ae7bdac7229093dee43cc138e44b2357a24f5f`. A capture must use the SFPI version required by its installed tt-metal, not automatically this latest SFPI commit.

## 1. Device-free frontend → C++ is demonstrated upstream

Exact official test:
[tt-lang/test/python/compiler_only_external_call.py](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/python/compiler_only_external_call.py).

Its lit requirement is `ttnn`, without `tt-device`. It runs:
```sh
env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=initial.mlir \
  python test/python/compiler_only_external_call.py > generated.log 2>&1
```

The test creates its arguments with:
```python
ttnn.from_torch(
    torch.zeros((64, 64), dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)
```
No device argument or open-device call is used. It then invokes the decorated operation and checks actual TTL IR and generated C++.

This demonstrates an offline harness pattern for the course matmul. It does **not** mean the unmodified tutorial host wrapper is hardware-free: the examples' wrappers open devices. Preserve the exact same matmul kernel definition, but use a dedicated host-only capture harness and skip output correctness checks when execution is disabled.

Useful capture flags from [compiler options](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/reference/compiler-options.md):
```sh
TTLANG_COMPILE_ONLY=1 \
TTLANG_INITIAL_MLIR=artifacts/initial.mlir \
TTLANG_FINAL_MLIR=artifacts/final-ttkernel.mlir \
TTLANG_VERBOSE_PASSES=1 \
TTLANG_DEBUG_LOCATIONS=1 \
TTLANG_VERBOSE_KERNELS=1 \
python capture_matmul.py > artifacts/compiler.log 2>&1
```
Here `capture_matmul.py` is a proposed harness name, not an existing verified script.

The Python implementation writes initial IR before its pass pipeline and final IR before the EmitC translation stage. Generated C++ is printed with its actual temporary file path. Preserve those files and exact options. Source: [ttl_api.py](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py).

Target selection nuance: that implementation derives `target_arch` from device-bearing inputs via `_device_target_arch`; host-only input compilation may leave it unspecified. Record that explicitly. Do not label a host-only generic lowering as a verified Blackhole-specific lowering merely because the eventual assembly toolchain is Blackhole.

## 2. Official offline Metalium C++ → ELF API

Header present at the **TT-Lang-pinned** tt-metal:
[experimental/offline_kernel_compile.hpp](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/api/tt-metalium/experimental/offline_kernel_compile.hpp).

Actual signature:
```cpp
void tt::tt_metal::experimental::CompileKernelOffline(
    const std::string& file_name,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const OfflineKernelCompileParams& params);
```

`OfflineKernelCompileParams` includes:
- `mode = AllSupportedProducts{}` (the only mode at this pin).
- Nonempty `output_dir`.
- `cb_compile_configs`: entries containing `uint8_t cb_index`, `DataFormat data_format`, and optional `Tile tile`.

`CBCompileConfigsFromProgram(program, kernel)` can derive CB metadata from a program without compiling it. Preserve all compile args/defines, math fidelity, destination width/sync options, and relevant include paths from the generated TT-Lang program. Supplying an arbitrary ComputeConfig or guessing CB indices would sever the evidence chain.

Implementation source:
[offline_kernel_compile.cpp](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/impl/experimental/offline_compile/offline_kernel_compile.cpp).
It generates the same JIT descriptors/wrappers and calls the JIT build machinery. It emits target ELFs under a kernel/compile-hash hierarchy. Compute kernels produce separate control-processor binaries. The public mode enumerates product configurations rather than taking a single simple `--arch blackhole` flag.

No Python binding or turnkey `ttlang --dump-asm` CLI was found in this audit. A C++ capture harness would need to call this API and use the installed Metalium build/package to link it. No verified standalone host link command was derived or executed.

Official hardware-free test evidence:
[test_offline_kernel_compile.cpp](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tests/tt_metal/tt_metal/api/test_offline_kernel_compile.cpp).
The `OfflineKernelCompileMockFixture` uses
`experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1)`, calls the offline API, and verifies that nonempty ELFs were emitted. This smoke test uses a reader kernel, not the course matmul; the public API also accepts ComputeConfig.

Important limitation from the same test: the offline path depends on precompiled firmware and does not build missing firmware itself. Tests skip when simulator/emulation environment selection changes the firmware build key. Do not combine offline compilation with `TT_METAL_SIMULATOR` and assume it is supported. Fabric-injected compile defines also need care.

## 3. Capture actual assembly and its matching compile commands

Official SFPI guidance:
[SFPI README](https://github.com/tenstorrent/sfpi/blob/21ae7bdac7229093dee43cc138e44b2357a24f5f/README.md).

Recommended settings:
```sh
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
export TT_METAL_KERNEL_MAP=1
export TT_METAL_LOGGER_LEVEL=info
export TT_METAL_LOGGER_TYPES=BuildKernels,LLRuntime
export TT_METAL_LOGGER_FILE=/absolute/capture/metal-logger.log
export TT_METAL_CACHE=/absolute/capture/metal-cache
```

These should be set before the actual TT-Metal process starts. A fresh capture cache avoids confusing prior binaries with newly generated ones. The existing `TT_METAL_FORCE_JIT_COMPILE=1` option is an alternative when retaining a cache is necessary.

Precise implementation:
[build.cpp](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/jit_build/build.cpp#L752).
- Lines 758–760: build-map mode adds saved temporaries and GCC dumps.
- Lines 813–814: the logging option prints `g++ compile cmd:`.
- Lines 874–879: build-map mode adds a linker map and saved link temporaries.
- Compile commands execute from the kernel's build directory. Relative includes resolve there, so retain that working directory in any replay.

Actual disassembly command, once the target ELF exists:
```sh
/path/to/matching/sfpi/compiler/bin/riscv-tt-elf-objdump -d \
  /absolute/capture/kernel.elf > /absolute/capture/kernel.disasm
```
These are parameterized paths, not paths validated on the course server. Use the matching SFPI tools and keep the ELF. The compiler's saved `.s` and linked-ELF disassembly answer different questions; link-time optimization can change final code.

For a standalone replay, start from the **exact logged command**, keep its macros/includes/generated wrappers, and preserve the exact cwd. Do not replace it with generic `g++ generated.cpp`: target CPU selection, firmware wrappers, CB descriptors, includes, compile args, and linker inputs matter.

## 4. Platform and simulator distinctions

TT-Lang's SFPI setup script supports Linux x86_64 and Linux aarch64, downloads the version recorded in installed TTNN metadata, verifies its hash, and extracts it beneath the environment:
[python/ttl/_setup/sfpi.py](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/_setup/sfpi.py).
No native macOS SFPI setup is exposed by that helper. The functional `tt-lang-sim` package runs on macOS but does not contain the compiler/device backend.

The compiler-backed `emule` workflow explicitly JITs **x86-64 shared objects**, requires its approved emulator repository, and uses a pinned Linux/amd64 Docker stack:
[simulator-getting-started.md](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/simulator-getting-started.md).
Its host machine code is not target Tensix/RISC-V assembly.

The separate public [ttsim](https://github.com/tenstorrent/ttsim) project models device instructions and can run TT-Metal without silicon; it is another possible later capture route, subject to compatible simulator/runtime versions. This research did not configure or run it.

## Future artifact manifest

A genuine matched capture should retain:
- frontend file and hash; selected shape/dtype/tiling; host harness;
- exact tt-lang, tt-metal, SFPI provenance and target selection;
- compiler flags and pass logs;
- initial, intermediate and final IR from that invocation;
- generated compute/reader/writer C++ and their hashes;
- all per-kernel configuration and CB metadata;
- actual compiler commands/cwds, generated wrappers/descriptors;
- target ELFs, saved assembly, disassembly and linker maps;
- clear validation status (compile-only, simulator, or silicon; correctness and timing separately).

Until then, any site snapshots must remain explicitly labeled illustrative placeholders. They are not authentic compiler dumps and must not be attributed to a course execution.

