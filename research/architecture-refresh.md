# Architecture chapter refresh — source audit

Verified live on 2026-10-16 (America/New_York). Research only; no hardware execution or benchmark validation. Recommendation: scope the detailed execution diagrams to **Wormhole B0 / Blackhole A0**, and distinguish architecture family, chip stepping, board SKU, and runtime-visible core grid.

## Generation taxonomy: use names and status, not an unsupported single count

A defensible teaching timeline contains **four named architecture families: Grayskull → Wormhole → Blackhole → Quasar**. This is a chronological list, not a claim that four generations are shipped or that every official source numbers generations identically.

| Family | Evidence-backed status | What to say in class |
|---|---|---|
| Grayskull | Official documentation home places it under Legacy and calls it the first-generation Tensix processor; e75/e150 hardware documentation remains available. | Historical first family; useful background, not the target assumed by these TT-Lang exercises. |
| Wormhole | Official n150/n300 product documentation and the ISA README identify shipped Wormhole B0 products. | A shipped family with a documented ISA; B0 is a stepping, not another generation. |
| Blackhole | Current firmware documentation calls its SoC third-generation; May 4, 2026 launch report says Blackhole Galaxy is shipping in volume. ISA README documents Blackhole A0 shipped in p100/p150. | A shipped family and suitable concrete target for this course's architectural examples. |
| Quasar | Current official ttsim README explicitly describes it as pre-silicon. Public binary simulator and software enablement exist; simulator implementation source is not public there. | The next named architecture in the public software ecosystem; do not describe it as a generally shipped fourth generation or imply this TT-Lang course has validated support. |

Primary evidence:
- [Official documentation home: hardware families and Legacy](https://docs.tenstorrent.com/index.html).
- [Grayskull e75/e150](https://docs.tenstorrent.com/aibs/grayskull/index.html).
- [Wormhole PCIe card documentation](https://docs.tenstorrent.com/aibs/wormhole/index.html).
- [Blackhole firmware overview: third-generation wording](https://docs.tenstorrent.com/tt-system-firmware/boards/tenstorrent/tt_blackhole/doc/index.html).
- [TT-Deploy, May 4 2026: Blackhole Galaxy shipping](https://tenstorrent.com/newsroom/tt-deploy).
- [Pinned ISA README: two documented architectures, WH B0/BH A0](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/README.md).
- [Pinned ttsim README: Quasar pre-silicon status](https://github.com/tenstorrent/ttsim/blob/464814f86583af9def3cf1643e58236ead759ae9/README.md).

**Ordinal inconsistency is real:** the [official July 18, 2024 Wormhole launch](https://tenstorrent.com/newsroom/tenstorrent-launches-next-generation-wormhole-based-developer-kits-and-workstations) quotes Blackhole as second generation, whereas current firmware documentation calls it third generation. Prefer family names and explicit status. If using a number, explain the counting convention. “The public ISA manual covers two architectures” is precise but is not a total-generation count.

Suggested site sentence: “This chapter places four named families on the timeline: legacy Grayskull, shipped Wormhole and Blackhole, and Quasar, which the current public simulator documentation identifies as pre-silicon. The detailed machine model below is for Wormhole B0 and Blackhole A0.”

## Concrete teaching model: one Tensix tile/core

For WH B0 and BH A0, distinguish the following:
- **Five Baby RISC-V control processors**, conventionally two overseeing data movement and three overseeing unpack, math, and pack. These are not five large matrix engines.
- **One Tensix coprocessor**, including two unpackers, one matrix unit/FPU, one vector unit/SFPU, one scalar unit/ThCon, and four packers.
- **Two NoC interfaces**, local L1 SRAM, and control/synchronization peripherals.
- SFPU is 32-wide SIMD over 32-bit lanes. Matrix and vector operations have different operand/precision paths; “everything runs on RISC-V scalar ALUs” is wrong.

Source: [Blackhole Tensix tile](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/BlackholeA0/TensixTile/README.md), [Wormhole Tensix tile](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/WormholeB0/TensixTile/README.md).

The three coprocessor instruction streams are controlled by RISC-V instruction dispatch. The compute backend is largely shared; three streams do not imply three independent matrix engines. Issuing a coprocessor instruction does not mean it has completed. Typical unpack/math/pack roles are a software convention, not an architectural restriction on which thread can issue to each backend unit. Source: [coprocessor execution and frontend](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/WormholeB0/TensixTile/TensixCoprocessor/README.md).

For TT-Lang, make the mapping explicit: operation → cooperating node programs; datamovement → copies/queues; compute → tile arithmetic and generated unpack/math/pack control. A Python compute function does not become one OS thread per tensor element.

## Memory hierarchy and useful distinctions

| Resource | Fact appropriate for class | Consequence |
|---|---|---|
| Host memory | Separate from accelerator storage | Host conversion/upload is not device kernel execution. |
| Device DRAM | Capacity storage reached through NoC transactions | Copy/placement/reuse decisions affect traffic. |
| Per-Tensix L1 | WH B0: **1464 KiB**; BH A0: **1536 KiB** | SRAM working set and queue capacity are finite; do not round both to an exact 1.5 MiB. |
| Operand/DST state | Specialized on-core compute storage | Fits in L1 does not imply fits in DST. |
| Baby RISC-V local RAM/cache | Separate control-processor resources | Do not say “there are no caches anywhere.” |

L1 is explicitly addressed scratchpad RAM, not a hardware-managed DRAM cache. Its capacity is not the maximum available to user buffers after code/runtime reservations. Ports and banks can contend. Source: [L1 specification](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/WormholeB0/TensixTile/L1.md).

Blackhole's Baby RISC-V cores have instruction caches and tiny 64-byte non-coherent L0 data caches, as well as per-processor local data RAM. Avoid transferring Wormhole private-memory limitations directly to Blackhole: Blackhole documents alternate addresses through which its local RAM is accessible to other processors/NoC. Source: [Blackhole Baby RISC-V memory map and caches](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/BlackholeA0/TensixTile/BabyRISCV/README.md).

For this course, show the common matrix path: L1 → unpack → source registers → matrix FPU → DST → pack → L1. SFPU processing operates on values through the DST/vector path. For a standard 32×32 tile, DST capacity is 4 FP32 tiles per active half with default double buffering, or 8 with full-sync mode; 16-bit mode doubles these counts. FP32 DST storage does not make every arithmetic operation IEEE FP32. Source: [compute engines and dataflow](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html).

## NoC and synchronization

Blackhole has two physically separate 2D torus NoCs connecting the same tile set in opposite directions. They carry asynchronous reads, writes, and atomics; addresses include the target endpoint and local address. A typical reader/NoC0 and writer/NoC1 assignment is a programming choice, not a read-only/write-only property of the networks. Blackhole flits are 512 bits, versus Wormhole's 256 bits, but do not derive real application bandwidth from this alone: headers, routing, congestion, banks, and ordering matter. Source: [Blackhole NoC](https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/BlackholeA0/NoC/README.md).

Teach two separate correctness questions:
1. Has the asynchronous payload copy completed?
2. Has the producer published the item / does the consumer own valid queue data?
A copy wait and circular-buffer publication answer different questions. Buffer depth permits overlap only if the program's schedule also allows it. The [Metalium guide](https://github.com/tenstorrent/tt-metal/blob/7c3dd90a675f0f00316786d01e4f60be42c3c9c0/METALIUM_GUIDE.md) supplies the reader/compute/writer model and the mapping of one compute source into three control-processor binaries.

## Tile layout

The standard teaching tile is **32×32 elements**, divided into four contiguous **16×16 faces**. In the standard orientation the faces are stored top-left, top-right, bottom-left, bottom-right; rows within each face are contiguous. This is not a simple row-major 32×32 array. BF16 tile = 2048 bytes, FP32 tile = 4096 bytes. Other shapes/formats exist, so say “standard tile in these exercises,” not “the only hardware tile.”

For row r, column c within a standard tile:
`offset_elements = ((r // 16) * 2 + (c // 16)) * 256 + (r % 16) * 16 + (c % 16)`.

Physical tile padding and logical validity masks are separate. Matmul padding must preserve the arithmetic (zero is suitable); ragged max/softmax masking requires the correct neutral/sentinel behavior rather than blindly treating every padded value as valid zero. Source: [official tile layout](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html), [pinned source](https://github.com/tenstorrent/tt-metal/blob/7c3dd90a675f0f00316786d01e4f60be42c3c9c0/docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst).

## Board counts and diagram cautions

Do not label a decorative 8×8 grid as the actual Blackhole layout. It should say schematic/example. Chip architecture maximum, board enabled cores, and runtime compute grid are different:
- Current Wormhole product page lists an 80-core processor, n150 cards with 72 cores, and n300 cards with 128 total (64 per ASIC).
- Current Blackhole card page lists 120 cores for p100a/p150a/p150b, while firmware describes an SoC with up to 140.
- A numeric Quasar simulator descriptor name is not proof of a marketed chip configuration.

Sources: [Wormhole card specifications](https://docs.tenstorrent.com/aibs/wormhole/index.html#card-comparison-table), [Blackhole card specifications](https://docs.tenstorrent.com/aibs/blackhole/index.html#card-comparison-table), [Blackhole firmware overview](https://docs.tenstorrent.com/tt-system-firmware/boards/tenstorrent/tt_blackhole/doc/index.html#overview).

For the course runner, record the actual available grid and software revision. Do not imply the learner can use every physical core, or assume all board SKUs expose the same grid.

## Scope of evidence

Pinned ISA commit: `8a34e25e04d1ad7dd129ac9231e126b85923fac3`.
Pinned tt-metal commit: `7c3dd90a675f0f00316786d01e4f60be42c3c9c0`.
Pinned ttsim commit: `464814f86583af9def3cf1643e58236ead759ae9` (commit timestamp 2026-10-17 UTC, which is still 2026-10-16 in America/New_York). README blob `ff9b5aadee8522109773b802f01bd44d7af37fdc`.

Public documentation reflects official claims and software contracts; it is not evidence that the course kernels have run on any architecture. Quasar simulator status is not a statement about whether private engineering prototypes exist.

