"""One-source teaching trace. All lowered representations are explicit placeholders."""
import textwrap
_trace_source=(ASSETS/'matmul.py').read_text()
_trace_compute=_trace_source[_trace_source.index('    @ttl.compute()'):_trace_source.index('    @ttl.datamovement()\n    def writer')]
TRACE_STAGES=[
    dict(label='Python',title='0 · Source and specialization',file='matmul.py · exact compute-kernel excerpt',
         code=textwrap.dedent(_trace_compute).strip(),
         description='<p>The source is <code>matmul(a,b,out)</code>, with A[64,128], B[128,96], and C[64,96]. Each queue has two slots, and each slot holds one 32×32 BF16 tile. The tile counts are therefore <code>Mt=2</code>, <code>Nt=3</code>, and <code>Kt=4</code>.</p><p>Follow output C_tile[1,2]. Its accumulator starts at zero and receives four tile products. Across all six outputs, each input queue supplies 24 blocks and the output queue supplies six. The frontend specializes the source using tensor metadata and captured configuration; tensor values remain runtime inputs.</p>',
         change='The frontend substitutes tensor metadata and captured configuration into the supported Python body. Tensor contents remain runtime inputs.',
         invariant='Six output tiles; four K terms per output; 24 input blocks on each input stream and six output blocks.',
         placeholder=False),
    dict(label='Initial TTL',title='1 · Python control flow and TTL IR',file='PLACEHOLDER · initial TTL sketch · abbreviated MLIR notation',
         code='''// PLACEHOLDER: illustrative structure, NOT a compiler dump.
// Each input block type is tensor<1x1x!ttcore.tile<32x32,bf16>>.
// Types/attributes/constant definitions omitted; ACC_T is unresolved.
scf.for %m = %c0 to %c2 step %c1 {
  scf.for %n = %c0 to %c3 step %c1 {
    %out = ttl.cb_reserve ...
    %zero = <zero block from ttl.block.fill>
    %final = scf.for %k = %c0 to %c4 step %c1
        iter_args(%acc = %zero) -> (<ACC_T>) {
      %a = ttl.cb_wait ...
      %b = ttl.cb_wait ...
      %product = ttl.matmul %a, %b ...
      %next = ttl.add %acc, %product ...
      ttl.cb_pop ...  // release A after its use
      ttl.cb_pop ...  // release B after its use
      scf.yield %next : <ACC_T>
    }
    ttl.store %final, %out ...
    ttl.cb_push ...
  }
}''',
         description='<p>In Python, <code>acc += ...</code> reassigns a variable. In SSA form, the K loop receives an accumulator value, computes its successor, and yields that value to the next iteration. After the fourth iteration, the loop result is stored in the output block. Replacing this recurrence with four independent stores would lose the earlier products.</p><p>The <code>with</code> scopes become buffer operations as well. Acquiring an input waits for a published block, and leaving its scope releases it. The output reservation spans the whole K loop, so publication occurs after the reduction. SSA names identify values; allocation passes later decide where to store them.</p>',
         change='Source constructs become typed operations, structured loops, SSA recurrence, and explicit dataflow transactions.',
         invariant='For C[1,2], the value yielded from iteration k is consumed by iteration k+1. It cannot be replaced by four independent overwrites.',
         placeholder=True),
    dict(label='Tile compute',title='2 · Block arithmetic to tile computation',file='PLACEHOLDER · compute/tile representation · explanatory notation',
         code='''// PLACEHOLDER: local recipe for ONE acquired A/B block pair.
// The outer source K loop still has four iterations.
local tile iteration extent = (m_local=1, n_local=1, k_local=1)
A map: (m_local,n_local,k_local) -> (m_local,k_local)
B map: (m_local,n_local,k_local) -> (k_local,n_local)
C map: (m_local,n_local,k_local) -> (m_local,n_local)
iterators = [parallel, parallel, reduction]

ttl.compute ... {
  %product = ttl.tile_matmul_block ...
  <accumulator update / tile store or yield, as selected by lowering>
}''',
         description='<p>One <code>@</code> in this source multiplies a single A tile by a single B tile. Its local recipe has one tile along each axis. The scalar dot products inside that tile have length 32; four iterations of the outer K loop cover scalar K=128.</p><p>The compiler forms compute regions and introduces tile indexing maps and iterator kinds. It also materializes intermediate values in buffers where their uses require storage. Throughout this process, the outer loops still describe six outputs, each with four K terms.</p><p>The separate 2×4 @ 4×3 block fixture in chapter 04 represents a larger block expression. It covers the same matrix dimensions with a different block size and comes from an independent compiler test.</p>',
         change='Block arithmetic gains tile indexing maps and parallel/reduction iteration structure; selected recipes become tile-level operations.',
         invariant='The global coordinates remain m∈[0,2), n∈[0,3), k∈[0,4). Local tile coordinates inside one block are a separate level.',
         placeholder=True),
    dict(label='Storage & sync',title='3 · Accumulator and queue storage',file='PLACEHOLDER · resource plan · physical assignments await capture',
         code='''PLACEHOLDER RESOURCE PLAN — NOT AN ALLOCATION DUMP
A queue:   1 tile/block × 2 blocks × 2,048 B = 4,096 B
B queue:   1 tile/block × 2 blocks × 2,048 B = 4,096 B
Out queue: 1 tile/block × 2 blocks × 2,048 B = 4,096 B
Declared queue payload:                         12,288 B

Accumulator across K:
  initial value: zero tile
  update: previous accumulator + A_tile @ B_tile
  storage strategy: <record actual DST or L1 decision>
  precision/configuration: <record actual compiler decision>

Physical CB indices/addresses: <capture later>
Compiler-created buffers:     <capture later>
DST slots and sync boundaries: <capture later>''',
         description='<p>The accumulator must survive all four K iterations. Recurrence analysis starts before tile conversion; this stage groups its storage consequences for discussion. A DST strategy and an L1 pack/reload strategy can differ in both traffic and numerical behavior. A real capture is needed to identify the choice for this kernel.</p><p>The three declared BF16 queues occupy 12 KiB per core. That total excludes compiler-created buffers, alignment, code, and runtime storage. Their logical names also do not determine physical CB indices or addresses.</p><p>One output tile fits within the usual four-tile FP32 double-buffered DST section. The allocator must still account for other live values and the selected configuration. Inputs remain live through their last use, and the writer must finish its output transfer before the output slot can be reused.</p>',
         change='Accumulation-scope lowering, DFB allocation, compute configuration, and DST assignment make lifetime and capacity decisions concrete. Their exact interleaving is shown in chapter 04.',
         invariant='Input storage remains live until its last use; output storage cannot be reused until writeback completes. The accumulator must survive all four updates.',
         placeholder=True),
    dict(label='TTKernel',title='4 · TTKernel operations and synchronization',file='PLACEHOLDER · TTKernel operation families · partial ordering sketch',
         code='''// PLACEHOLDER: representative operation families for this source.
reader:
  ttkernel.cb_reserve_back ...        // obtain input capacity
  <NoC reads for the selected A and B tiles>
  <read completion before publication>
  ttkernel.cb_push_back ...

compute:
  ttkernel.cb_wait_front ...          // acquire each input
  <DST acquire / accumulator restoration at selected boundaries>
  ttkernel.matmul_block ...           // one local tile product
  <lower the accumulator update and preserve its state>
  <commit / wait / pack / release at selected boundaries>
  ttkernel.cb_pop_front ...           // release consumed input

writer:
  ttkernel.cb_wait_front ...          // completed output only
  <NoC write and write completion>
  ttkernel.cb_pop_front ...

// Positions/counts of tile_regs_* and pack calls require actual IR.
// Output publication remains once per completed K reduction.''',
         description='<p>At this stage, buffer operations use circular-buffer APIs and arithmetic uses compute APIs such as <code>matmul_block</code>. Source copy waits become transfer-completion operations; scope exits become queue publication or release; output stores require materialization and packing.</p><p>DST synchronization coordinates math and pack. Acquisition grants ownership, commit/wait establishes the handoff, and release permits reuse. The selected accumulation strategy determines where these operations appear, so the sketch leaves those boundaries unresolved. Input and output block counts alone cannot tell us how many internal pack or register-synchronization calls will be emitted.</p><p>Check the transfer boundaries too: the reader completes both input copies before publishing, and the writer completes its copy before releasing the output slot.</p>',
         change='High-level TTL abstractions become target-facing buffer, NoC, compute, and register-synchronization operations. Initialization and pack-combining passes refine them afterward.',
         invariant='The reader completes both input copies before publishing; the writer completes its outbound copy before releasing the output slot.',
         placeholder=True),
    dict(label='EmitC / C++',title='5 · EmitC and device C++',file='PLACEHOLDER · C++ pseudocode · authored teaching outline',
         code='''// PLACEHOLDER: source-level loop structure and API families only.
// Definitions, arguments, config, and exact accumulator path omitted.
void compute_kernel() {
  for (unsigned m = 0; m < 2; ++m) {
    for (unsigned n = 0; n < 3; ++n) {
      cb_reserve_back(/* output CB, one tile */);
      /* initialize accumulator using the selected storage strategy */
      for (unsigned k = 0; k < 4; ++k) {
        cb_wait_front(/* A CB, one tile */);
        cb_wait_front(/* B CB, one tile */);
        /* selected initialization, DST synchronization and restore */
        matmul_block(/* actual generated operands and tile arguments */);
        /* selected accumulation / materialization */
        cb_pop_front(/* A CB, one tile */);
        cb_pop_front(/* B CB, one tile */);
      }
      /* pack final result and finish its destination handoff */
      cb_push_back(/* output CB, one tile */);
    }
  }
}''',
         description='<p>EmitC represents calls, types, expressions, and control flow that can be translated to C++. The resulting source is compiled with the target libraries and configuration. The outline shows the original m, n, and k loops and representative API families; their exact arguments and accumulator implementation await a real capture.</p><p>Reader and writer produce separate generated source, while compute code supports the unpack, math, and pack control roles. Keep these identities when collecting the corresponding binaries.</p><p>The device compiler may inline calls, fold constants, or unroll loops. Follow the output values and their dependencies across these changes: the implementation must still produce six output tiles from four K terms each.</p>',
         change='Typed target operations become source-language expressions and calls. The later device compiler performs another round of optimization and code generation.',
         invariant='The same six output tiles and four-term recurrence must be implemented, even if loops are folded or unrolled.',
         placeholder=True),
    dict(label='Assembly',title='6 · Device instructions',file='PLACEHOLDER · symbolic assembly outline · no ELF or disassembly captured',
         code='''# PLACEHOLDER: NOT real disassembly; no addresses or encodings.
# One possible unrolled-or-looped K region, shown as a loop here.
# <...> marks unresolved code and register placeholders.

    <initialize k register to 0 and bound register to 4>
.L_k:
    <input wait / readiness implementation>
    <accumulator restore and DST synchronization, if required>
    <Tensix matrix-engine instruction-issue sequence>
    <accumulator update / preservation>
    <input-release implementation>
    addi <k_reg>, <k_reg>, 1
    blt  <k_reg>, <bound_reg>, .L_k
    <final packing and output publication>

# addi/blt illustrate scalar RISC-V loop control only.
# Actual code may use different branches or unroll the loop.
# Matrix instructions and exact synchronization await target capture.''',
         description='<p>The displayed <code>addi</code> and <code>blt</code> illustrate scalar RISC-V loop control. The matrix product is issued to the Tensix compute machinery in the unresolved region. Its actual instructions, registers, addresses, and synchronization sequence depend on the target binary.</p><p>With a matching ELF, select a short region and map it back to generated C++ and the accumulator path. Identify the processor role and distinguish instruction issue from the wait or dependency that establishes completion. The compiler may also unroll the K loop, so the loop shown here is only one possible structure.</p><p>No ELF was captured for this walkthrough. Instruction counts and timing must come from a later compilation and execution, with the source, shape, flags, toolchain, and target recorded together.</p>',
         change='The device toolchain lowers generated C++ and library code into processor-specific machine code; disassembly decodes a particular resulting ELF.',
         invariant='A report must associate its binary with this source, shape, flags, target, and processor role. A schematic cannot substitute for that evidence.',
         placeholder=True)
]
for _s in TRACE_STAGES:
    _s['provenance']=('<strong>PLACEHOLDER — authored teaching sketch; no compilation performed.</strong>' if _s['placeholder'] else '<a href="downloads/matmul.py">Exact course source excerpt</a> · source review only; compilation pending.')

MATMUL_TRACE=note('About these examples','The Python excerpt is taken from <code>matmul.py</code>. The IR, storage plans, C++, and assembly are illustrative placeholders based on that source and shape. They explain the lowering, but have not been produced by a compiler run.','note')
MATMUL_TRACE+='<p>We will follow C_tile[1,2]. The table lists its four input pairs and their linear tile IDs. Each source-level block contains one tile.</p>'+table(['K iteration','A tile / linear tile ID','B tile / linear tile ID','Output'],[[str(k),'A[1,'+str(k)+'] / '+str(4+k),'B['+str(k)+',2] / '+str(3*k+2),'Accumulate into C[1,2] / 5'] for k in range(4)])
MATMUL_TRACE+='<div class="stage-line">'+''.join('<span>'+s['label']+'</span>'+('<i>→</i>' if i<len(TRACE_STAGES)-1 else '') for i,s in enumerate(TRACE_STAGES))+'</div>'
for i,s in enumerate(TRACE_STAGES):
    MATMUL_TRACE+='<h3 id="trace-step-'+str(i)+'">'+s['title']+'</h3>'+code(s['code'],s['file'])+s['description']
MATMUL_TRACE+=refs(('Accumulation and pass ordering',UP+'lib/Dialect/TTL/Pipelines/TTLPipelines.cpp'),('One-tile matmul lowering test',UP+'test/ttlang/Conversion/TTLToTTKernel/matmul_block_lowering.mlir'),('Compute formation and lifetime rules',UP+'docs/development/ComputeOpCreation.md'))
MATMUL_TRACE+='<h3>Checks at each stage</h3>'+table(['Invariant','How to check it','What would be a bug'],[['Indexing','For selected output tile 5, A IDs are 4,5,6,7 and B IDs are 2,5,8,11.','Changing reader traversal without matching consumer order.'],['Recurrence','Zero → acc₁ → acc₂ → acc₃ → acc₄ → one output store.','Dropping a K term, overwriting the accumulator, or publishing a partial result.'],['Lifetime','Copy complete before publish; last read before release; output write complete before reuse.','Packing or transferring through storage that has already been recycled.']])
MATMUL_TRACE+='<h3>Capturing real artifacts later</h3><p>Keep the source and its SHA-256, exact tensor properties and compiler options, initial IR, named per-pass checkpoints, final MLIR, generated kernel files, processor-specific ELF files, matching disassembly, and toolchain/target versions together. All capture-only fields in the supplied manifest are intentionally empty.</p><div class="actions"><a class="button-secondary" href="downloads/matmul-trace-placeholders.txt" download>↓ Annotated placeholder walkthrough</a><a class="button-secondary" href="downloads/matmul-capture-manifest.json" download>↓ Capture manifest template</a></div>'
