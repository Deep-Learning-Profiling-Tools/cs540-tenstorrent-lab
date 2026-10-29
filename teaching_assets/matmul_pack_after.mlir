// Q7 teaching excerpt: packing region of matmul_2x4_4x3.
// Manually transcribed from the upstream CHECK/COMBINED expectations and
// pack-combining implementation; this is not a captured compiler dump.
// Computation and register acquisition are omitted. For reading only.
// Adapted from Tenstorrent TT-Lang (Apache-2.0; see LICENSE-tt-lang.txt).

!output_cb = !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>

func.func @matmul_2x4_4x3_pack_excerpt(%cb2: !output_cb)
    attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  ttkernel.tile_regs_wait() : () -> ()
  "ttkernel.pack_tile_block"(%c0, %cb2, %c6)
      : (index, !output_cb, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  return
}
