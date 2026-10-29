overview=header('00 / Lab overview','Tenstorrent programming lab','In this lab, you will write Tenstorrent kernels in TT-Lang and study how the compiler lowers them. Start with matrix multiplication, then implement and measure vector addition, ragged sum, and ragged softmax.',[])
overview+=section('path','Contents','<div class="path-list">'+''.join('<a class="path-row" href="#'+r+'"><span class="step">'+str(i).zfill(2)+'</span><div><b>'+t+'</b><p>'+d+'</p></div><span class="arrow">→</span></a>' for i,(r,t,d) in enumerate([
('architecture','Architecture','Generations, Tensix cores, DRAM, NoC, L1, unpack, math, and pack.'),('tt-lang','TT-Lang','Operations, nodes, blocks, queues, copies, and the official language APIs.'),('matmul','Matmul example','Single-core code, larger blocks, and multicore code.'),('compiler','IR & assembly','Compilation stages, example IR and C++, and inspection tools.'),('vector-add','Practice 1 · Vector addition','Elementwise addition, a PyTorch baseline, and ten test shapes.'),('ragged-sum','Practice 2 · Ragged sum','Sum each row over its valid prefix.'),('ragged-softmax','Practice 3 · Ragged softmax','Normalize each row over its valid prefix.'),('report','Analysis & report','Timing, implementation, optimization and profiling, and eight IR questions.')],1))+'</div>')
overview+=section('workflow','Course workflow','<ol><li>Study architecture and matmul, then inspect the compiler artifacts.</li><li>Adapt the example kernels to the problem’s submission interface. Kernels receive preallocated device tensors and write output arguments.</li><li>Submit code on the '+source('course OJ',OJ)+'.</li><li>The platform handles compilation, device setup, execution, and timing. For analysis, use the results returned by the platform and compiler artifacts supplied by the instructor.</li></ol>')
lesson('overview','Lab overview','Lab contents and submission workflow.',overview)

ISA='https://github.com/tenstorrent/tt-isa-documentation/blob/8a34e25e04d1ad7dd129ac9231e126b85923fac3/'
ISA_LIVE='https://github.com/tenstorrent/tt-isa-documentation/blob/main/'
METAL_ADV='https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/'
WH_CARDS='https://docs.tenstorrent.com/aibs/wormhole/index.html#card-comparison-table'
BH_CARDS='https://docs.tenstorrent.com/aibs/blackhole/index.html#card-comparison-table'
arch=header('01 / Architecture','Tenstorrent architecture','A Tenstorrent accelerator contains many Tensix cores, each with local memory and compute engines. This chapter follows data from the host through device DRAM, the on-chip network, and one core.',[])
arch+=section('generations','Architecture generations',
    '<p>The four architecture families covered here are <strong>Grayskull, Wormhole, Blackhole, and Quasar</strong>.</p>'+
    table(['Family','Hardware documentation'],[
        ['Grayskull','First-generation Tensix family; retained in the legacy hardware documentation.'],
        ['Wormhole','PCIe cards include n150 and n300. The public ISA manual describes the B0 stepping.'],
        ['Blackhole','PCIe cards include p100a, p150a, and p150b. The public ISA manual describes the A0 stepping.'],
        ['Quasar','The official simulator repository describes Quasar as pre-silicon and provides simulator binaries.']])+
    refs(('Grayskull hardware','https://docs.tenstorrent.com/aibs/grayskull/specifications.html'),('ISA manual',ISA_LIVE+'README.md'),('Quasar status','https://github.com/tenstorrent/ttsim#distribution')))
arch+=section('identities','Board specifications',
    '<p>These published specifications describe particular cards. The n300 contains two accelerator chips; its memory and bandwidth entries below are board totals. GB/s denotes the advertised peak DRAM bandwidth.</p>'+
    table(['Specification','Wormhole n150','Wormhole n300','Blackhole p100a','Blackhole p150a'],[
        ['Accelerator chips','1','2','1','1'],
        ['Tensix cores','72','128 (64/chip)','120','120'],
        ['AI clock','1 GHz','1 GHz','Up to 1.35 GHz','Up to 1.35 GHz'],
        ['GDDR6 capacity','12 GB','24 GB (12/chip)','28 GB','32 GB'],
        ['DRAM bandwidth','288 GB/s','576 GB/s (288/chip)','448 GB/s','512 GB/s'],
        ['Host interface','PCIe 4.0 ×16','PCIe 4.0 ×16','PCIe 5.0 ×16','PCIe 5.0 ×16'],
        ['Total board power','160 W','300 W','300 W','300 W']])+
    refs(('Wormhole card specifications',WH_CARDS),('Blackhole card specifications',BH_CARDS)))
arch+=section('device-tools','Querying and managing devices',
    '<p>Run these commands on the accelerator host. <strong>TT-SMI</strong> lists attached devices and reports board identity, firmware, PCIe links, clocks, temperature, and power.</p>'+
    code("""tt-smi --version           # Installed tool version
 tt-smi --list              # Device IDs, PCI addresses, and board models
 tt-smi                     # Interactive status and telemetry
 tt-smi --snapshot          # Current information as structured output
 tt-smi --snapshot --filename device-snapshot.json
 lspci -d 1e52:             # Tenstorrent devices enumerated by Linux PCIe""".replace('\n ', '\n'),'Shell · device queries')+
    '<p>Match the board model to the specification table. Record the firmware version and negotiated PCIe link speed/width; use the telemetry view to observe clock, temperature, and power while a workload runs.</p>'+
    table(['Tool','Purpose'],[
        [source('TT-SMI','https://github.com/tenstorrent/tt-smi'),'Device discovery, telemetry, firmware versions, and board resets.'],
        [source('TT-Flash','https://github.com/tenstorrent/tt-flash'),'Install or verify device firmware bundles.'],
        [source('TT-Topology','https://github.com/tenstorrent/tt-topology'),'Configure Ethernet routing across supported multi-card systems.'],
        [source('TT-Toplike','https://github.com/tenstorrent/tt-toplike'),'Live monitoring of power, temperature, clocks, and host processes.']])+
    '<p>Resets, firmware updates, and topology changes are administrator tasks on the shared course machines. Use the query commands above to inspect the devices. Run <code>tt-smi --help</code> for the options supported by the installed version.</p>'+refs(('TT-SMI commands and fields','https://github.com/tenstorrent/tt-smi#usage')))
arch+=section('machine','The multicore device',
    '<p>The host CPU prepares inputs and launches work through PCIe. GDDR6 memory holds device tensors. A <strong>network on chip (NoC)</strong> carries data between memory controllers, Tensix cores, and other endpoints. Each Tensix core has its own L1 SRAM and arithmetic engines.</p>'+CHIP_FIGURE+
    '<p>A matrix can be divided into output blocks and distributed across the cores. Each core fetches the input blocks it needs into its local memory, computes its assigned outputs, and transfers the results back to DRAM. Cores can also transfer data directly to one another.</p>'+
    refs(('Metalium architecture introduction',LAB+'lab1/lab1.html#introduction-to-tenstorrent-architecture'),('Blackhole NoC',ISA_LIVE+'BlackholeA0/NoC/README.md')))
arch+=section('engines','Inside one Tensix core',
    '<p><strong>“Baby RISC-V” means a small control CPU.</strong> Tenstorrent uses the nickname for five compact, 32-bit processors inside each Tensix core: B, NC, T0, T1, and T2. Each executes a program with instructions, registers, loops, and branches. Their design emphasizes low area and power.</p>'+
    '<p><strong>Packers, unpackers, and math engines are separate execution units.</strong> The RISC-V processors issue commands to them. For example, T2 runs the control code that tells a packer which destination tile to format and write into L1; the packer performs the transfer. T1 issues arithmetic commands to the matrix engine or vector engine (SFPU), which perform the numerical work.</p>'+
    '<p>The <strong>matrix engine (FPU)</strong> multiplies and accumulates tiles. The <strong>vector engine (SFPU)</strong> performs elementwise and special-function operations. Both use <strong>destination registers (DST)</strong> to hold compute results. Unpackers bring operands from L1 into compute registers; packers format results and write them to L1.</p>'+CORE_FIGURE+
    table(['Resource per core (Wormhole B0 / Blackhole A0)','Count and function'],[
        ['Small RISC-V control CPUs (“Baby”)','5 programmable processors: B, NC, T0, T1, and T2. Execute control code and issue commands to the engines.'],
        ['Unpackers','2 engines for operand loading and format conversion.'],
        ['Matrix engine / FPU','1 engine for matrix arithmetic.'],
        ['Vector engine / SFPU','1 engine; 32 SIMD lanes, each 32 bits wide.'],
        ['Scalar unit / ThCon','1 unit; integer scalar operations and 128-bit L1 memory operations.'],
        ['Packers','4 engines for writing compute results to L1.'],
        ['NoC interfaces','2 connections per core.']])+
    '<p>B and NC run the data-movement kernels: they issue NoC transfers between L1, DRAM, and other cores. T0 issues unpack instructions; T1 issues matrix and vector math instructions; T2 issues pack instructions. The dashed arrows show this usual software assignment. The hardware provides shared execution resources, so the assignment is established by the kernel software.</p>'+
    refs(('Why they are called Baby RISC-V',ISA_LIVE+'BlackholeA0/TensixTile/BabyRISCV/README.md'),('Command issue and execution',ISA_LIVE+'WormholeB0/TensixTile/TensixCoprocessor/README.md'),('Control-processor roles',ISA_LIVE+'WormholeB0/TensixTile/BabyRISCV/README.md'),('Blackhole Tensix core',ISA_LIVE+'BlackholeA0/TensixTile/README.md'),('Wormhole Tensix core',ISA_LIVE+'WormholeB0/TensixTile/README.md'),('Compute dataflow',METAL_ADV+'compute_engines_and_dataflow_within_tensix.html')))
arch+=section('memory','Memory hierarchy',
    '<p><strong>L1 is software-managed scratchpad SRAM.</strong> Each core keeps its current operands, intermediate buffers, and code there. The unpack and pack engines move tensor values between L1 and the compute registers.</p>'+MEMORY_FIGURE+
    table(['Storage','Wormhole B0','Blackhole A0'],[
        ['L1 SRAM per Tensix core','1,464 KiB = 1,499,136 bytes','1,536 KiB = 1,572,864 bytes'],
        ['BF16 tile payload','2 KiB for 32×32 values','2 KiB for 32×32 values'],
        ['Available L1 buffer budget','Program- and runtime-dependent','Program- and runtime-dependent'],
        ['Destination registers (DST) per core','32 KiB total','32 KiB total'],
        ['Full DST, 16-bit elements','16,384 values = 16 standard tiles','16,384 values = 16 standard tiles'],
        ['Full DST, 32-bit elements','8,192 values = 8 standard tiles','8,192 values = 8 standard tiles']])+
    '<p><strong>Available buffer space</strong> is the L1 region left for a program after firmware, code, and runtime reservations. The compiler places input queues, output queues, intermediate buffers, and communication state within that budget. Existing allocations and alignment reduce the space left for another buffer.</p><p>TT-Lang checks the combined per-core allocation with <code>ttl-validate-cb-budget</code> and subsequent lowering checks. The target supplies the default budget; <code>--ttl-l1-budget N</code> overrides it in bytes. Use the budget configured for the course runner when sizing buffers. Increasing this option only changes the compiler limit; physical memory and runtime reservations stay the same.</p><div class="formula">remaining budget = configured L1 buffer budget − planned allocations\nBF16 tile payloads that fit ≤ floor(remaining budget / 2,048)</div><p>The tile calculation is an upper bound before any additional alignment or placement constraints. DST has its own separate 32 KiB register capacity.</p>'+
    refs(('TT-Lang L1 budget checks',UP+'docs/sphinx/reference/compiler-options.md#ttl-validate-cb-budget'),('L1 capacities',ISA_LIVE+'WormholeB0/TensixTile/L1.md'),('Destination register organization',ISA_LIVE+'WormholeB0/TensixTile/TensixCoprocessor/Dst.md'),('Blackhole destination registers',ISA_LIVE+'BlackholeA0/TensixTile/TensixCoprocessor/Dst.md')))
arch+=section('noc','Network on chip (NoC)',
    '<p>A NoC is the communication network inside the accelerator. Routers forward packets between endpoints. An endpoint may expose a core’s L1 memory, a DRAM controller, or another device resource. A transfer identifies both the destination endpoint and an address within it.</p>'+
    '<p>Wormhole and Blackhole have two physically separate NoCs. Each forms a <strong>two-dimensional torus</strong>: rows and columns connect back to their opposite edges. The two networks connect the same endpoints and send packets in opposite directions.</p>'+NOC_FIGURE+
    '<p>A DRAM read returns data to the requesting core’s L1. A core-to-core write sends bytes from one L1 to another. Multicast sends a payload to several destination cores. The destination must have free storage, and consumers wait for transfer completion before reading the received data.</p>'+
    '<p>Packets travel in fixed-size units called <strong>flits</strong>. Blackhole doubles the flit width compared with Wormhole.</p>'+
    table(['Network property','Wormhole B0','Blackhole A0'],[
        ['Physical NoCs','2','2'],
        ['Packet flit size','256 bits = 32 bytes','512 bits = 64 bytes'],
        ['Transfer operations','Reads, writes, atomics','Reads, writes, atomics']])+
    refs(('Wormhole NoC',ISA_LIVE+'WormholeB0/NoC/README.md'),('Blackhole NoC and official topology figures',ISA_LIVE+'BlackholeA0/NoC/README.md')))
arch+=section('tile-storage','Tile layout in DRAM and L1',
    '<p>This layout describes the <strong>bytes of a tiled tensor buffer</strong> in device DRAM or L1 SRAM. Copying an intact tile between these memories preserves its byte order. The unpacker converts stored values into the representation used by compute registers.</p>'+
    '<p>A standard tile covers 32×32 scalar values. Its four 16×16 faces are stored in sequence: top-left, top-right, bottom-left, bottom-right. Within each face, values follow row-major order.</p>'+TILE_FIGURE+
    '<div class="formula">face = 2 × (r // 16) + (c // 16)\nelement_offset = 256 × face + 16 × (r % 16) + (c % 16)\nbyte_offset = element_offset × bytes_per_element</div>'+
    '<p>For BF16, value (0,16) has element offset 256 and byte offset 512. A 32×32 FP32 tile uses the same face order and occupies 4,096 bytes. Tilization converts a row-major tensor into this storage order; untilization reverses the conversion.</p>'+
    refs(('Official tile layout and address diagrams',METAL_ADV+'tiles.html')))
arch+=section('dst','Destination registers (DST)',
    '<p><strong>DST</strong> is the destination register storage inside a Tensix core. Matrix operations accumulate results there. The SFPU reads and writes these values for vector operations. Packers read the completed results and write them to L1.</p>'+
    '<p>The 32 KiB store holds 16 tiles of 16-bit values or 8 tiles of 32-bit values. Double buffering splits it into two sections: math works on one while pack drains the other. Synchronization transfers ownership between these phases.</p>'+DST_FIGURE+
    table(['Element width','One double-buffered section (16 KiB)','Full-sync section (32 KiB)'],[
        ['16 bits','8 tiles of 32×32 values','16 tiles of 32×32 values'],
        ['32 bits','4 tiles of 32×32 values','8 tiles of 32×32 values']])+
    '<p>For example, a 2×3 tile output block needs six destination tiles. With 32-bit destinations and double buffering, the four-tile section is too small to hold the whole block at once. The compiler schedules smaller groups or uses additional storage. The IR & assembly chapter explains how the compiler schedules these groups.</p>'+
    refs(('DST synchronization and configuration',METAL_ADV+'compute_engines_and_dataflow_within_tensix.html#dst-register'),('DST physical storage',ISA_LIVE+'WormholeB0/TensixTile/TensixCoprocessor/Dst.md')))
lesson('architecture','Architecture','Board specifications, multicore organization, Tensix engines, memory, networks, and tile storage.',arch)
