// Q7 teaching excerpt: packing region of matmul_2x4_4x3.
// Manually transcribed from the upstream CHECK/COMBINED expectations and
// pack-combining implementation; this is not a captured compiler dump.
// Computation and register acquisition are omitted. For reading only.
// Adapted from Tenstorrent TT-Lang (Apache-2.0; see LICENSE-tt-lang.txt).

!output_cb = !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>

func.func @matmul_2x4_4x3_pack_excerpt(%cb2: !output_cb)
    attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb2, %c0, true)
      : (index, !output_cb, index) -> ()
  ttkernel.pack_tile(%c1, %cb2, %c1, true)
      : (index, !output_cb, index) -> ()
  ttkernel.pack_tile(%c2, %cb2, %c2, true)
      : (index, !output_cb, index) -> ()
  ttkernel.pack_tile(%c3, %cb2, %c3, true)
      : (index, !output_cb, index) -> ()
  ttkernel.pack_tile(%c4, %cb2, %c4, true)
      : (index, !output_cb, index) -> ()
  ttkernel.pack_tile(%c5, %cb2, %c5, true)
      : (index, !output_cb, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  return
}
