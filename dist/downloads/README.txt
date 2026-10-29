CS540 TT-Lang Lab — Tutorial materials

Read the website or reading-guide.html for the full tutorial.
Kernels import ttl, receive preallocated tiled device tensors, write output
arguments, and return None. They are not standalone host launchers and cannot
be called with ordinary NumPy/Torch tensors. Use the instructor's TT-Lang runner.

matmul.py: single core, scalar dimensions divisible by32.
matmul_larger.py: larger blocks on one core; dimensions divisible by64.
matmul_blocked.py: blocked/full-grid adaptation; default dimensions divisible
by64. Pad inputs with zeros and crop outputs in the runner.
pytorch_baselines.py: FP32 reference functions for all three exercise definitions.
Exercise inputs and outputs use FP32; row lengths use int64.
vector_add.py: optional low-level example using packed physical storage.
ragged_baselines.py: optional low-level examples using a mask tensor and padded
storage; these use a different interface from the logical PyTorch references.
Their full-row implementation supports padded width<=256.
Long ragged cases require a streaming extension with bounded buffers.

Teaching kernels have been source-reviewed and Python syntax-checked. They await
compilation and execution on course hardware. Pin and validate the course image.
The course OJ still needs a compatible TT-Lang runner and problem setup.

CPU semantics: python check_references.py (requires NumPy).
References check the mathematical contract; they are not accelerator kernels.
matmul-trace-placeholders.txt: annotated teaching sketches; compiler capture pending.
matmul-capture-manifest.json: source/shape contract with capture fields left null.
Compiler .mlir files are authentic upstream test inputs with expected patterns. The .h file contains inline assembly from
the Blackhole library. See provenance.json and licenses.
