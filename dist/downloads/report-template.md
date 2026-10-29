# CS540 · TT-Lang lab report

Name:
GMU ID:

Submit the completed report as a separate PDF.

## 1. Time reporting — 20 points

- Use figures or tables to report execution times for all 10 cases of each exercise. Label the exercise, case, implementation, time unit, and submission ID.
- Compare your initial correct implementation with your final implementation. Report speedup as initial time / final time, using the same timing scope. Label missing or failed results.
- Use the measurements returned by the submission platform and state what its reported time includes.

[Your response]

## 2. Implementation sketch — 20 points

- For each exercise, give concise pseudocode or a diagram of your final implementation.
- Describe work assignment, data movement, and computation. Explain how the implementation handles irregular sizes, empty rows, and singleton rows where applicable.

[Your response]

## 3. Optimization and profiling metric analysis — 20 points

- For each exercise, explain at least one change from the initial implementation, the cost it targets, and its effect on execution time across cases.
- Analyze the profiling metrics returned by the platform. Give each metric’s name, unit, scope, and before/after values, and explain how it supports or challenges your hypothesis.
- Discuss a regression or a case with little improvement. If a relevant metric is unavailable, identify it and explain what it would help determine; separate measured results from hypotheses.

[Your response]

## 4. IR analysis — 40 points

Answer all eight questions. Each is worth 5 points out of 100. Use the supplied MLIR files and linked sources; cite file names and line numbers. No compiler or accelerator access is required.

- [matmul_to_compute.mlir](matmul_to_compute.mlir)
- [matmul_block_lowering.mlir](matmul_block_lowering.mlir)
- [matmul_pack_before.mlir](matmul_pack_before.mlir)
- [matmul_pack_after.mlir](matmul_pack_after.mlir)

### Q1. Host and device compilation paths — 5 points

Provide a table of the compilation and execution stages, in sequence, from a TT-Lang Python operation call to execution on the accelerator. Use four columns: stage, execution hardware (CPU or accelerator), input, and output.



[Your analysis]

### Q2. Tensor types and dimensions — 5 points

In matmul_to_compute.mlir, inspect matmul_2x4_4x3. Write the full types, tile dimensions, and number of tiles of the two input tensors and the result.



[Your analysis]

### Q3. Iteration space and indexing maps — 5 points

For matmul_2x4_4x3 in matmul_to_compute.mlir, explain the three affine indexing maps for A, B, and the output.



[Your analysis]

### Q4. Circular-buffer capacities — 5 points

In matmul_2x4_4x3 in matmul_block_lowering.mlir, identify the three ttl.bind_cb declarations and their block_count values. Calculate each buffer’s capacity in tiles and BF16 payload bytes.



[Your analysis]

### Q5. Register and packing sequence — 5 points

Explain the CHECK expectations for matmul_1x1_bf16 in matmul_block_lowering.mlir. What are we checking?



[Your analysis]

### Q6. Reduction loop after lowering — 5 points

In the CHECK expectations for matmul_2x4_4x3 in matmul_block_lowering.mlir, give the scf.for lower and upper bounds.



[Your analysis]

### Q7. Pack-combining transformation — 5 points

Compare the supplied matmul_pack_before.mlir and matmul_pack_after.mlir files. How many packing operations are present before and after the transformation? You do not need to run a compiler.

[Before transformation](matmul_pack_before.mlir) · [After transformation](matmul_pack_after.mlir)

[Your analysis]

### Q8. Circular-buffer lifetimes — 5 points

In matmul_add_accumulator in matmul_block_lowering.mlir, identify the input and output circular buffers. Explain the roles of ttl.cb_wait, ttl.cb_reserve, ttl.cb_push, and ttl.cb_pop.

[Read the supplied MLIR file](matmul_block_lowering.mlir)

[Your analysis]

