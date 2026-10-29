lang=header('02 / TT-Lang','Programming with TT-Lang','A one-tile addition example shows how host code launches cooperating device kernels.',[])
lang+=section('operation','Start with a program skeleton',
    '<p>We will add two 32×32 BF16 tiles. The host prepares three device tensors and calls one operation. That operation contains a reader, a compute kernel, and a writer.</p>'+code("""import ttl

# DEVICE DEFINITION — compiled by TT-Lang
@ttl.operation(grid=(1, 1))
def add_one_tile(a, b, out):
    # Declare three local queues: a_q, b_q, out_q.
    ...

    @ttl.datamovement()
    def reader():
        ...  # a, b → a_q, b_q

    @ttl.compute()
    def compute():
        ...  # a_q + b_q → out_q

    @ttl.datamovement()
    def writer():
        ...  # out_q → out

# HOST INVOCATION — device tensors supplied by the runner
add_one_tile(a_device, b_device, out_device)
""",'Program skeleton · omitted bodies are expanded below')+
    '<p>The decorated definition is written in Python and processed on the host. TT-Lang compiles its kernel bodies into device programs. The call at the bottom launches the operation; the host calls <code>add_one_tile</code> once, and the device kernels cooperate through the queues.</p>'+
    table(['Host program','Device program'],[['Prepare inputs and device tensors','Reader transfers inputs into local queues.'],['Invoke <code>add_one_tile(a, b, out)</code>','Compute adds acquired input blocks.'],['Wait, read back, and check the result','Writer transfers the result to the output tensor.']]))

lang+=section('host','Host code: prepare, invoke, and check',
    '<p>The host is the CPU process running the platform’s server-side runner. It initializes the accelerator, prepares the input values, allocates tiled device tensors, and uploads the inputs. It also allocates the output tensor before the operation is called.</p>'+code("""# Host workflow — pseudocode for course-runner services
initialize the assigned accelerator
prepare a_host and b_host: shape (32, 32), dtype BF16
allocate a_device, b_device, out_device: tiled device storage
upload a_host → a_device and b_host → b_device

# Actual TT-Lang invocation
add_one_tile(a_device, b_device, out_device)

wait for device completion
read out_device back into host row-major storage
compare with a_host + b_host using the required tolerance
release device resources
""",'Server-side host workflow · pseudocode')+
    '<p><code>a_device</code>, <code>b_device</code>, and <code>out_device</code> are handles to device tensors. Their shape, format, and placement describe the storage the operation will use. <code>add_one_tile</code> writes into <code>out_device</code> and returns <code>None</code>.</p>'+
    '<p>For this example, both inputs contain exactly 1,024 values, so no padding is needed. The host reference uses the same BF16-rounded inputs. The runner waits for completion before reading the result or stopping an execution timer.</p>')

lang+=section('specialization','Host invocation: compilation and specialization','<p>The runner passes preallocated tensors to an operation. Shape, dtype, layout, and memory properties determine its specialization; tensor values are supplied at runtime. Calls with the same specialization can reuse the compiled program with new tensor values.</p><p>Block sizes, grid dimensions, and other scalar settings are compile-time values captured from an enclosing scope or factory function. Changing a block size therefore selects a new compiled program. The examples pass tensors as runtime arguments and capture scalar settings in the enclosing scope.</p>'+table(['Level','Example','Responsibility'],[['Host runner','Allocate tensors, invoke operation, check outputs','Device setup, transfers, validation, and timing boundaries.'],['Operation definition','Declare DFBs and kernels','Shared resources and execution structure.'],['Device kernel body','Loop over tile coordinates, acquire blocks, copy or compute','Supported DSL syntax lowered by the compiler.']])+'<p>The course runner handles host setup and invokes the operation. Importing a kernel file defines it; invocation triggers compilation and execution as needed.</p>')

lang+=section('language-model','Device code: the operation and its queues',
    '<p><code>@ttl.operation(grid=(1, 1))</code> selects one execution node. The outer function declares resources shared by the nested kernels. Each <code>@ttl.datamovement()</code> body describes transfer and synchronization work; the <code>@ttl.compute()</code> body describes arithmetic on blocks.</p>'+code("""@ttl.operation(grid=(1, 1))
def add_one_tile(a, b, out):
    a_q = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_q = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_q = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    # The three kernel definitions below are nested here.
""",'Device definition · shared buffers')+
    '<p>A <strong>dataflow buffer (DFB)</strong> is a local queue of blocks. Here <code>shape=(1, 1)</code> means one 32×32 tile per block, and <code>block_count=2</code> gives the queue two slots. The corresponding tensor supplies the element format. Each BF16 slot holds 2,048 payload bytes.</p>'+
    '<div class="formula">a, b → reader → a_q, b_q → compute → out_q → writer → out</div>'+
    '<p>The reader produces blocks for compute. Compute produces blocks for the writer. The kernels can progress independently when their input blocks and output slots are available. Their definition order in the source establishes the program structure; queue dependencies determine when work can proceed.</p>')

lang+=section('reader','Device reader: bring in the operands',
    code("""@ttl.datamovement()
def reader():
    with a_q.reserve() as a_blk, b_q.reserve() as b_blk:
        a_request = ttl.copy(a[0, 0], a_blk)
        b_request = ttl.copy(b[0, 0], b_blk)
        a_request.wait()
        b_request.wait()
""",'Inside add_one_tile · reader kernel')+
    '<p><code>reserve()</code> waits for a free queue slot and gives the reader a writable block. <code>a[0, 0]</code> selects the first tile of the device tensor. <code>ttl.copy</code> starts a transfer into the reserved block and returns a request; <code>request.wait()</code> waits for that transfer to finish.</p>'+
    '<p>Leaving each reserve scope publishes its completed block. Compute can then acquire it. Both requests are issued before either wait, allowing the two input transfers to overlap.</p>')

lang+=section('compute','Device compute: add the blocks',
    code("""@ttl.compute()
def compute():
    with a_q.wait() as a_blk, b_q.wait() as b_blk:
        with out_q.reserve() as out_blk:
            out_blk.store(a_blk + b_blk)
""",'Inside add_one_tile · compute kernel')+
    '<p><code>a_q.wait()</code> and <code>b_q.wait()</code> acquire published input blocks. Reserving <code>out_q</code> obtains space for the result. The expression <code>a_blk + b_blk</code> describes elementwise addition of all 1,024 values, and <code>store</code> materializes the result in the output block.</p>'+
    '<p>Closing the output scope publishes the result. Closing the input scopes releases those slots so the reader can reuse them. TT-Lang lowers this body into the required unpack, math, and pack work on the Tensix core.</p>')

lang+=section('writer','Device writer: store the result',
    code("""@ttl.datamovement()
def writer():
    with out_q.wait() as out_blk:
        ttl.copy(out_blk, out[0, 0]).wait()
""",'Inside add_one_tile · writer kernel')+
    '<p>The writer acquires a completed block from <code>out_q</code>, copies it into the output device tensor, and waits for the copy to finish. Leaving the scope releases the queue slot. The result is now in <code>out</code>; the host runner handles completion and readback.</p>'+
    '<p>The two uses of <code>wait</code> refer to different events: <code>out_q.wait()</code> waits for a published block, while the request returned by <code>ttl.copy</code> waits for a transfer to complete.</p>')

lang+=section('first-operation','Complete device operation','<p>Putting the declarations and three kernel bodies together gives the operation below. The host invokes it with the three preallocated tensors from the skeleton.</p>'+code('''import ttl

@ttl.operation(grid=(1, 1))
def add_one_tile(a, b, out):
    a_q = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_q = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_q = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def reader():
        with a_q.reserve() as a_blk, b_q.reserve() as b_blk:
            a_request = ttl.copy(a[0, 0], a_blk)
            b_request = ttl.copy(b[0, 0], b_blk)
            a_request.wait()
            b_request.wait()

    @ttl.compute()
    def compute():
        with a_q.wait() as a_blk, b_q.wait() as b_blk:
            with out_q.reserve() as out_blk:
                out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def writer():
        with out_q.wait() as out_blk:
            ttl.copy(out_blk, out[0, 0]).wait()
''','Teaching adaptation · one-tile addition')+'<ol><li>The reader reserves A and B storage, starts both copies, and waits for their bytes before scope exit publishes them.</li><li>Compute acquires the input blocks and materializes their sum in the reserved output. Closing the scopes publishes output and releases inputs.</li><li>The writer waits for output, completes its outbound copy, and then releases the slot.</li></ol><p>The declared buffers occupy <code>3 × 2 × 2,048 = 12,288</code> payload bytes per node. This example needs only one slot per queue. We use two so the declarations can be reused when adding a loop and overlapping successive transfers.</p>'+refs(('Official elementwise progression',DOC+'elementwise-tutorial/index.html'),('Block and copy semantics',UP+'docs/sphinx/specs/TTLangSpecification.md')))

lang+=section('queue','Queue ownership: reserve, publish, acquire, release','<p><code>make_dataflow_buffer_like(a,shape=(1,1),block_count=2)</code> allocates scratch storage for two blocks on each participating node. The tensor supplies format information. Queue depth limits how far the producer can run ahead.</p>'+table(['Call','Role','Meaning'],[['<code>dfb.reserve()</code>','Producer','Wait for capacity; acquire the next write block. Existing bytes are not guaranteed zero.'],['<code>block.push()</code>','Producer','Publish a completed block.'],['<code>dfb.wait()</code>','Consumer','Wait for a published block; acquire the next read block.'],['<code>block.pop()</code>','Consumer','Release the block and return capacity.']])+code('''with input_dfb.reserve() as tile:
    transfer = ttl.copy(a[0, 0], tile)
    transfer.wait()  # bytes have arrived
# reserve scope exit pushes the block

with input_dfb.wait() as tile:
    ...  # consume the block here
# wait scope exit pops the block''')+'<p>The wait completes the transfer, and leaving the reserve scope publishes the block. Both steps are needed before compute can consume the input. On an output queue, releasing a block before writeback completes would allow its storage to be overwritten too soon.</p>')

lang+=section('tiles','Tensor, tile, block, and grid shapes',table(['Object','Example','Meaning'],[['Logical tensor','<code>A.shape=(64,128)</code>','64 scalar rows and128 scalar columns.'],['Tile grid','<code>(2,4)</code>','Eight32×32 tiles cover A.'],['DFB block','<code>shape=(1,2)</code>','One acquired block covers32×64 scalars.'],['Launch grid','<code>grid=(2,2)</code>','Four execution nodes, each with local buffers.']])+'<p>In a data-movement kernel, <code>a[m,k]</code> indexes tiles. For example, <code>a[0:2,1:3]</code> selects a 2×2 tile block covering scalar rows 0–63 and columns 32–95. The <code>a.shape</code> property uses scalar dimensions. This difference matters when converting loop bounds into slice indices.</p><p>A 32×32 tile occupies 2,048 payload bytes in BF16 or 4,096 in FP32. Its four 16×16 faces are stored contiguously, as described in the architecture chapter. The host runtime performs tilization before the device kernel uses these tiles.</p>'+note('Shape and storage transformations','Loop tiling changes traversal. Tilization changes storage order. Padding changes physical extents. A kernel may require all three.'))

lang+=section('padding','Padding requirements','<p>For scalar extent D, tile padding gives <code>Dp=32×ceil(D/32)</code>. Matmul padding along K must contribute zero to the dot product; output outside the logical result is cropped. Blocked kernels may require alignment to32×BM,32×BN,32×BK, stricter than one tile.</p>'+hint('Check A[65,97] × B[97,33]','Tile-only padding yields A[96,128], B[128,64], C[96,64], with six output tiles. For2×2 output blocks and BK=2, pad M to128 instead. Zero-fill the input extensions and crop C to[65,33].'))

lang+=section('overlap','Asynchrony and double buffering','<p><code>ttl.copy</code> returns a transfer request. Issuing the A and B requests before calling <code>request.wait()</code> allows their transfers to overlap, subject to the memory system and the chosen routes.</p><p>With two slots, the reader can fill block k+1 while compute consumes block k. With one slot, it must wait for compute to release that storage. Additional slots consume more L1 and help only while there is work available to overlap. All kernels on a core must agree on the stream order and block count.</p>'+hint('Why does a queue not carry matrix coordinates?','The stream order encodes the coordinates. In the baseline, the reader produces (m,n,k) order and compute consumes it in the same order. Reordering one side alone pairs the wrong operands. A missing production or extra wait can hang.'))

lang+=section('capacity','L1 and DST requirements','<p>For BF16 input blocks BM×BK and BK×BN, output BM×BN, and queue depth d, the three declared queues use:</p><div class="formula">L1 payload = 2,048 × d × (BM×BK + BK×BN + BM×BN) bytes</div><p>This estimate covers the three declared queue payloads. Alignment, compiler-created buffers, code, and runtime resources also use L1. Check the compiler’s allocation and budget diagnostics for the complete requirement.</p><p>DST has a smaller capacity than L1. Double-buffered 32-bit DST storage provides four 32×32 tiles per section. Full synchronization exposes eight tiles with less overlap. Consequently, a block that fits in L1 may still need DST subblocking.</p>'+refs(('Compute configuration and DST',UP+'docs/sphinx/reference/compiler-options.md')))

lang+=section('node-ownership','Work assignment','<p><code>ttl.node(dims=1)</code> gives a flattened logical node index; <code>ttl.grid_size(dims=1)</code> gives the number of nodes. A simple assignment gives node p the block IDs <code>p, p+P, p+2P, …</code>. Every kernel on that node must use the same ownership rule and traversal order.</p><p>For Q=10 blocks and P=4 nodes, the assignments are [0,4,8], [1,5,9], [2,6], and [3,7]. This covers each block once, but workloads with unequal block costs can still be imbalanced. Logical coordinates are mapped by the runtime; do not use them as raw NoC coordinates.</p><p>For 2D grids, name the axes explicitly. The official matmul tutorial uses grid coordinates ordered as N then M. Swapping those meanings changes which output rectangle each node owns.</p>'+refs(('Grid and node APIs',DOC+'tour/index.html#grid-and-node-functions'),('Full-grid matmul source',UP+'examples/matmul-tutorial/step_4_multinode_grid_full.py')))

lang+=section('communication','Local queues and cross-node pipes','<p>A DFB connects kernels on one node. A <code>ttl.Pipe</code> describes a connection between nodes, and <code>ttl.PipeNet</code> groups these connections into a communication pattern. A multicast receiver still needs to reserve space before the sender transfers a block.</p><p>Cross-node transfers add dependencies between the sender and receivers; each receiver publishes a block after its transfer completes.</p>'+refs(('Pipe specification',DOC+'specs/TTLangSpecification.html#pipe'),('2D multicast example',UP+'examples/metal_examples/2d_mcast_matmul/ttlang/2d_mcast_matmul.py')))

lang+=section('api','API reference for the examples',table(['API','Units / result','Purpose'],[['<code>@ttl.operation(grid=...)</code>','Tuple or tutorial <code>"full"</code> grid','One cooperating operation.'],['<code>@ttl.datamovement()</code>','Device data-movement kernel','Slices, transfers, requests, handoffs.'],['<code>@ttl.compute()</code>','Device compute kernel','Block arithmetic and math.'],['<code>ttl.grid_size(dims=1)</code>','Flattened node count','Partition work.'],['<code>ttl.node(dims=1)</code>','Logical node index','Choose output ownership using a logical node index.'],['<code>make_dataflow_buffer_like</code>','Shape in tiles for tiled operands','Declare local scratch queue capacity.'],['<code>ttl.copy(...).wait()</code>','Request then completion','Finish DMA before publish/reuse.'],['<code>ttl.block.fill</code>','Block expression','Explicit initialization.'],['<code>a_blk @ b_blk</code>','Block matrix product','Reduce over scalar K and K tiles.'],['<code>out_blk.store(expr)</code>','Materialize expression','Write acquired output; scope exit publishes.'],['<code>ttl.math.reduce_sum / reduce_max</code>','Explicit dims and result shape','Reduce selected scalar axes across tiles.'],['<code>ttl.block.broadcast</code>','Explicit dims and target shape','Replicate reduced values.']]))

lang+=section('documentation','Further reading',table(['Official resource','Use it for'],[[source('Tour of TT-Lang',DOC+'tour/index.html'),'First encounter with operations, nodes, blocks, and producer/consumer structure.'],[source('Language specification',DOC+'specs/TTLangSpecification.html'),'Precise shape units, ownership states, tensor slices, copies, pipes, and supported features.'],[source('Matmul tutorial',DOC+'matmul-tutorial/index.html'),'Follow the custom-kernel steps from one tile to blocks and multiple nodes.'],[source('Compiler options',DOC+'reference/compiler-options.html'),'Specialization settings, math configuration, debug flags, and pass reference.']]))

lesson('tt-lang','TT-Lang','A code-first walkthrough of host setup, device kernels, queues, and TT-Lang APIs.',lang)
