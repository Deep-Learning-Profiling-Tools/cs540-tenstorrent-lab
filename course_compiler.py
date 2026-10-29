PIPELINE=TRACE_STAGES
comp=header('04 / IR & assembly','Compiler stages and inspection tools','Use the compiler tools to follow a kernel from TT-Lang source to device code.',[])
comp+=section('pipeline','Compilation stages',
    '<div class="stage-line">'+''.join('<span>'+x+'</span>'+('<i>→</i>' if i<5 else '') for i,x in enumerate(['TT-Lang Python','TTL IR','TTKernel IR','EmitC IR','Kernel C++','Device ELF']))+'</div>'
    +table(['Stage','Purpose'],[
        ['Frontend → TTL IR','Specialize the Python operation and represent its computation, data movement, and control flow in MLIR.'],
        ['TTL IR → TTKernel IR','Lower block operations and dataflow into operations for device kernel APIs.'],
        ['TTKernel IR → EmitC IR → C++','Express the lowered program as C++ calls and generate device-kernel source.'],
        ['C++ → device ELF','Compile and link the generated kernels with the Tenstorrent device toolchain. The resulting ELF files contain executable code for the assigned device processors.']])
    +'<p>IR dumps, generated C++, and disassembly show different points along this path. Disassembly is obtained by decoding a device ELF.</p>'
    +refs(('TT-Lang compiler pipeline',DOC+'reference/compiler-options.html#pipeline-ttl-to-ttkernel-pipeline'),('Device compilation','https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp')))
comp+=section('examples','Example IR and C++',
    '<p>A single-tile matmul multiplies two 32×32 BF16 tiles. This shortened TTL excerpt comes from an upstream lowering test; buffer declarations and the surrounding function are omitted.</p>'
    +code('''%mm = ttl.matmul %a, %b
    : tensor<1x1x!ttcore.tile<32x32, bf16>>,
      tensor<1x1x!ttcore.tile<32x32, bf16>>
   -> tensor<1x1x!ttcore.tile<32x32, bf16>>
ttl.store %mm, %reserve
    : tensor<1x1x!ttcore.tile<32x32, bf16>>,
      tensor<1x1x!ttcore.tile<32x32, bf16>>''','TTL IR · excerpt from a compiler test')
    +'<p><code>%a</code> and <code>%b</code> are input values; <code>%mm</code> is the result. The outer <code>1x1</code> counts tiles, and <code>32x32</code> gives the elements in each tile. <code>ttl.store</code> writes the result into the reserved output block.</p>'
    +code('''// Illustrative C++ excerpt; buffer IDs have descriptive names.
// Hardware initialization and queue operations are omitted.
tile_regs_acquire();
matmul_block_init(cb_a, cb_b, 0, 1, 1, 1);
matmul_block(cb_a, cb_b, 0, 0, 0, 0, 1, 1, 1);
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_out, 0);
tile_regs_release();''','Kernel C++ · illustrative adaptation')
    +'<p>The lowered code makes register ownership, matrix-engine work, and output packing explicit. The final three arguments of <code>matmul_block</code> are the output columns, output rows, and inner dimension in tiles; all are one here. <code>pack_tile</code> writes the result from destination registers to the output circular buffer. This excerpt illustrates the API calls; it has not been captured from a compilation of the tutorial kernel.</p>'
    +refs(('One-tile lowering test',UP+'test/ttlang/Conversion/TTLToTTKernel/matmul_block_lowering.mlir'),('Matmul API','https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/matmul_block.html')))
comp+=section('tools','Inspection tools',
    '<h3>Compiler options in Python</h3>'
    +'<p>TT-Lang provides compiler options that you can set in Python scripts. List the available options with:</p>'
    +code('''from ttl.compiler_options import CompilerOptions

print(CompilerOptions.usage())''','Python · list compiler options')
    +'<p>Pass a string of options to <code>@ttl.operation</code>:</p>'
    +code('''import ttl

@ttl.operation(grid=(1, 1), options="--no-ttl-maximize-dst")
def matmul(a, b, out):
    # Reader, compute, and writer definitions go here.
    ...''','Python · operation configuration; kernel body omitted')
    +'<p>You can also supply <code>options=</code> when calling an existing operation:</p>'
    +code('''# a_device, b_device, and out_device are prepared device tensors.
matmul(
    a_device, b_device, out_device,
    options="--no-ttl-maximize-dst --no-ttl-combine-pack-tiles",
)''','Python · options for one invocation')
    +'<p>The call-time string replaces the decorator’s option string for that invocation. Environment and command-line settings can override these values.</p>'
    +'<h4>Common compiler options</h4>'
    +table(['Option','Default','Effect'],[
        ['<code>--ttl-maximize-dst</code>','Enabled','Group tile work to use destination registers efficiently. Use <code>--no-ttl-maximize-dst</code> to compare per-tile synchronization.'],
        ['<code>--ttl-fpu-binary-ops</code>','Enabled','Allow eligible add, subtract, and multiply operations to use the matrix engine. <code>--no-ttl-fpu-binary-ops</code> selects the vector engine.'],
        ['<code>--ttl-combine-pack-tiles</code>','Enabled','Combine consecutive output-packing operations. Use <code>--no-ttl-combine-pack-tiles</code> to disable this transformation.'],
        ['<code>--ttl-subblock-sync</code>','Disabled','Synchronize individual subblocks within a larger block.'],
        ['<code>--ttl-accumulation-strategy auto</code>','<code>auto</code>','Choose accumulation storage automatically; <code>dst</code> and <code>l1-pack</code> select destination registers and L1-backed accumulation, respectively.']])
    +'<p>The Python compiler also reads these environment variables:</p>'
    +table(['Environment variable','Purpose'],[
        ['<code>TTLANG_INITIAL_MLIR=initial.mlir</code>','Write MLIR before the compiler pipeline to this file.'],
        ['<code>TTLANG_FINAL_MLIR=final.mlir</code>','Write MLIR after the compiler pipeline to this file.'],
        ['<code>TTLANG_VERBOSE_PASSES=1</code>','Print IR after each compiler pass.'],
        ['<code>TTLANG_DEBUG_LOCATIONS=1</code>','Include source locations in printed MLIR.']])
    +refs(('Python option parser',UP+'python/ttl/compiler_options.py'),('Compiler options',DOC+'reference/compiler-options.html'))
    +'<h3><code>ttlang-opt</code></h3>'
    +'<p>The standalone MLIR optimizer reads an MLIR file and runs passes. It accepts standard MLIR driver options, including the two IR-printing flags below.</p>'
    +table(['Option','Purpose'],[
        ['<code>--help</code>','List the options and passes available in this installation.'],
        ['<code>-p \'ttl-to-ttkernel-pipeline\'</code>','Run the registered TT-Lang lowering pipeline.'],
        ['<code>--mlir-print-ir-before-all</code>','Print IR before each pass.'],
        ['<code>--mlir-print-ir-after-all</code>','Print IR after each pass.'],
        ['<code>-o lowered.mlir</code>','Write the resulting IR to a file.']])
    +refs(('ttlang-opt reference',DOC+'reference/compiler-options.html#ttlang-opt-pass-reference'),('MLIR IR printing','https://mlir.llvm.org/docs/PassManagement/#ir-printing'))
    +'<h3><code>ttlang-translate</code></h3>'
    +'<p>Translate prepared kernel IR into C++ source.</p>'
    +table(['Option','Purpose'],[
        ['<code>--ttkernel-to-cpp</code>','Select the kernel-to-C++ translation.'],
        ['<code>-o kernel.cpp</code>','Write the C++ source to a file.']])
    +'<h3>Device binary tools</h3>'
    +'<p>Use the executables supplied by the matching Tenstorrent SFPI toolchain.</p>'
    +table(['Tool','Options','Purpose'],[
        ['<code>riscv-tt-elf-readelf</code>','<code>-h</code><br><code>-S</code><br><code>-s</code>','Show the ELF header, section headers, and symbol table, respectively.'],
        ['<code>riscv-tt-elf-objdump</code>','<code>-d</code>','Disassemble executable sections of the device ELF.']])
    +refs(('Tenstorrent toolchain','https://github.com/tenstorrent/sfpi'))
    +'<h3>Comparing compiler output</h3>'
    +table(['Tool','Purpose'],[
        ['<code>FileCheck</code>','Check compiler output against patterns in a test file.'],
        ['<code>diff</code>','Compare two IR or source files.'],
        ['<code>rg</code>','Find an operation or symbol in saved artifacts.']])
    +refs(('FileCheck','https://llvm.org/docs/CommandGuide/FileCheck.html')))
comp+=section('commands','Example commands',
    '<p>The following examples illustrate the tool interfaces. Input files and tool paths stand for artifacts and tools from the same compiler installation.</p>'
    +code('''# Print the IR after each pass in a registered pipeline.
ttlang-opt input.mlir -p 'ttl-to-ttkernel-pipeline' \\
  --mlir-print-ir-after-all -o lowered.mlir 2> passes.log

# Translate IR prepared for C++ emission.
ttlang-translate --ttkernel-to-cpp emitc.mlir -o kernel.cpp''','Compiler tools')
    +code('''# Inspect one processor binary with its matching toolchain.
/path/to/sfpi/compiler/bin/riscv-tt-elf-readelf -h -S -s kernel.elf
/path/to/sfpi/compiler/bin/riscv-tt-elf-objdump -d kernel.elf > kernel.disasm''','Binary tools')
    +'<p>Use filenames and processor labels to keep related artifacts together. A reader, writer, and compute program can produce separate binaries.</p>')
lesson('compiler','IR & assembly','High-level compilation stages, an IR/C++ example, and inspection tools.',comp)
