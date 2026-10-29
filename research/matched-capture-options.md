# Actual matched matmul capture: options investigated

Research stopped when the user explicitly chose placeholders for now. No runtime
was installed, no compiler was run, no remote system was mutated, and no actual
matmul IR/C++/ELF artifacts were generated in this investigation.

TT-Lang source pin: `a47484458bd964a6b92c9e4698aeb922585f08cc`.
Its actual tt-metal submodule pin is `0e9d200db976120c129ab0deb13aa3f6d972b723`.
Downloaded source evidence is under `research/matched-capture-sources/`.

## Quickest supported frontend capture without a device

The official test `test/python/compiler_only_external_call.py` proves a
hardware-free compilation path: it requires the runtime Python package, not a
device, constructs **host-resident tiled tensor containers** using `from_torch`
with no `device=` argument, and calls a TT-Lang operation with
`TTLANG_COMPILE_ONLY=1`. It captures actual initial MLIR and generated C++.

For the course, reuse the existing matmul kernel unchanged and put this host
setup only in the capture harness. Do not expose this runtime package as a
second instructional API. The compiler-only host setup should not open a device,
upload tensors, execute numerical checks, or claim accelerator execution.

Conceptual host harness (adaptation, not executed):

```python
import os
os.environ["TTLANG_COMPILE_ONLY"] = "1"
import torch
import ttnn  # capture-harness tensor container only
from matmul import matmul

def host_tensor(shape):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

matmul(host_tensor((64,128)), host_tensor((128,96)), host_tensor((64,96)))
```

Set `TTLANG_INITIAL_MLIR`, `TTLANG_FINAL_MLIR`, `TTLANG_VERBOSE_PASSES=1`, and
`TTLANG_VERBOSE_KERNELS=1` when invoking that harness. The final dump is after
EmitC conversion at this revision; the before/after pass log supplies precise
same-compilation snapshots.

Target caveat: `_device_target_arch` returns None for host inputs. That path is
valid for compiler tests but does not establish Blackhole-specific lowering.
The private `_compile_kernel` function accepts `target_arch="blackhole"`; the
public operation decorator does not expose that argument. An eventual capture
could use that internal API with explicit provenance or run a standalone
pipeline with the architecture attribute supplied to saved initial IR, recording
the latter as a distinct compiler invocation. This was not tested.

Sources:
- [Official no-device compiler test](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/test/python/compiler_only_external_call.py)
- [Host target detection](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L758)
- [Internal compiler function](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/ttl_api.py#L3168)

## Real device ELF without physical hardware: official offline API

The pinned tt-metal has an experimental C++ API:

```cpp
tt::tt_metal::experimental::CompileKernelOffline(
    const std::string& source_path,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const OfflineKernelCompileParams& params);
```

`OfflineKernelCompileParams` contains a nonempty output directory and per-CB
compile configuration (`cb_index`, `DataFormat`, optional `Tile`). The currently
supported mode is `AllSupportedProducts`: it enumerates multiple Wormhole and
Blackhole product/build configurations instead of accepting one simple target.
The implementation constructs a local HAL and BuildEnvManager, deliberately
avoids device/MetalContext singletons, invokes the real JIT build machinery and
copies processor ELF files into the requested output tree. The official
`CPU_CompileKernelOfflineEmitsExpectedSubtreeForReaderKernel` test validates ELF
emission without physical hardware using a mock fixture.

An eventual matched path is:

1. Compile the one Python matmul into actual IR and generated C++.
2. Preserve each generated kernel's exact compile arguments, defines, processor
   role, data formats, tile metadata and compute configuration.
3. Feed those sources/configurations to the offline compiler using the matching
   full Metal installation and SFPI toolchain.
4. Select and label the resulting target-specific reader/writer/compute ELFs.
5. Disassemble with that toolchain's `riscv-tt-elf-objdump -d`, recording hashes
   and provenance. These would be real compiled artifacts, not execution proof.

Important constraints: this API is C++, experimental, and needs the matching
Metal library/source resources, precompiled firmware and SFPI compiler. No Python
binding or turnkey CLI was confirmed. The test explicitly skips offline compile
under simulator/emulator mode because its runtime build key lacks matching
precompiled firmware; run the offline path in ordinary non-simulator mode.
Fabric defines require additional live-topology information and are unnecessary
for this simple local matmul exercise.

Sources:
- [Offline compilation API](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/api/tt-metalium/experimental/offline_kernel_compile.hpp)
- [Hardware-free implementation](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/impl/experimental/offline_compile/offline_kernel_compile.cpp)
- [Official CPU-only ELF emission test](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tests/tt_metal/tt_metal/api/test_offline_kernel_compile.cpp)
- [Enumerated offline targets](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/jit_build/jit_device_config.cpp)

## Installation options and current environment observations

At lookup time, PyPI reported `tt-lang==1.1.6` with one
`cp312-cp312-manylinux_2_38_x86_64` wheel, ~72 MB, requiring `ttnn==0.74.0`.
The package version is not proof of matching the course source pin; record
`ttl.build_info()` and change the artifact's documented revision if using it.
The pinned source documentation also describes S3 nightly/light wheels and
prebuilt dist/ird Docker images, but no exact matching wheel was selected.

The local computer reports `arm64`; docker, podman, colima and ttlang-opt were
not found on PATH. The portable `tt-lang-sim` package is functional simulation
only, not a compiler. A Linux x86_64 Python3.12 host is the quickest plausible
place to install the public compiler package for IR/C++ capture. The full
offline-ELF route may require a newer/matching Metal build than that package.

`tt-lang-setup-sfpi` reads the installed runtime's `sfpi-version`, downloads the
matching SFPI release archive, verifies SHA-256 and extracts inside the venv.
It supports Linux x86_64/aarch64, not native macOS. It does not by itself create
kernel binaries.

Sources:
- [Installation guide](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/getting-started.md)
- [Public compiler package](https://pypi.org/project/tt-lang/1.1.6/)
- [SFPI installer implementation](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/python/ttl/_setup/sfpi.py)

## Emulator route: useful but not sufficient for requested assembly

The official compiler-backed `emule` backend uses Linux/amd64 Docker, a pinned
Metal stack, and an approved emulator repository URL supplied by its owners.
Installation builds TT-Lang and runtime dependencies. The stack supports an
emulated Blackhole P150 target, but the emulator JITs **x86-64 shared objects**.
Therefore emulated execution alone does not produce the requested RISC-V device
assembly. Do not publish host x86 disassembly as accelerator disassembly.

The native Python simulator generates neither actual compiler IR nor device
assembly. Both simulator routes must be labeled honestly if used later.

Sources:
- [Compiler-backed emulation prerequisites and limitations](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/docs/sphinx/simulator-getting-started.md)
- [Pinned emulation stack](https://github.com/tenstorrent/tt-lang/blob/a47484458bd964a6b92c9e4698aeb922585f08cc/config/tt-lang-emule-stack.json)

For the user's current placeholder choice, none of these proposed capture paths
should be described as completed. Illustrative snapshots should remain labeled
as authored placeholders, with consistent dimensions/source and no generated-
artifact, tested-kernel, or measured-performance claims.
