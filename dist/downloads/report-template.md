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

Answer all eight questions. Each is worth 5 points out of 100. Use the named functions and CHECK expectations in the supplied compiler tests; cite file names and line numbers. These questions can be answered from the linked sources and compiler tests without running a kernel.

- [matmul_to_compute.mlir](matmul_to_compute.mlir)
- [matmul_block_lowering.mlir](matmul_block_lowering.mlir)

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

In the CHECK expectations for matmul_1x1_bf16 in matmul_block_lowering.mlir, list the operations from tile_regs_acquire through tile_regs_release in order. Identify the matrix operation and the operation that writes the result to the output buffer. Explain the roles of commit and wait in this sequence.



[Your analysis]

### Q6. Reduction loop after lowering — 5 points

In the CHECK expectations for matmul_2x4_4x3 in matmul_block_lowering.mlir, give the scf.for lower and upper bounds. Using the input-index expressions stated in the fixture, list the tile-index pairs passed to matmul_block over the K iterations. Give the ct, rt, and kt arguments, and identify which register-ownership operations remain outside the loop.



[Your analysis]

### Q7. Pack-combining transformation — 5 points

Compare CHECK and COMBINED for matmul_2x4_4x3 in matmul_block_lowering.mlir. Name the extra pass in the second RUN line. Count the packing operations before and after it, name their operation types, and state the destination-register and output-buffer tile ranges they cover. Explain why the 1x1 case keeps its existing packing operation.



[Your analysis]

### Q8. Commands for inspecting IR — 5 points

Using the IR & assembly chapter, answer the following questions (1 point each). Give the option or command and briefly explain its output. You do not need to run the commands.

1. Which environment variable saves the initial MLIR from a TT-Lang Python invocation?
2. Which environment variable saves the final MLIR from that invocation?
3. Which environment variable prints IR after each pass in the Python compiler pipeline?
4. Which environment variable includes source locations in printed MLIR?
5. Write a ttlang-opt command that runs ttl-to-ttkernel-pipeline on input.mlir and prints IR before and after every pass.

[Your analysis]

