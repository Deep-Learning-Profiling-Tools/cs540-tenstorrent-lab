// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __riscv_xtttensixbh || (__clang__ && defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_TRISC))
#define TTI_INSN(ENCODING) ({ __asm__ __volatile__(".ttinsn %0" ::"n"(unsigned(ENCODING))); })
#define TT_INSN(ENCODING)  void(::ckernel::instrn_buffer[0] = unsigned(ENCODING))
#elif defined(ARCH_BLACKHOLE) && defined(LLK_BOOT_BRISC)
// The llk test infra uses TTI macros on brisc and somehow executes
// it. So icky. See #58141
#define TTI_INSN(ENCODING) ({ __asm__ __volatile__(".4byte %0" ::"n"(unsigned((((ENCODING) >> 30) & 3) | (((ENCODING) & 0x3fffffff) << 2)))); })
#define TT_INSN(ENCODING)  ({ __asm__ __volatile__(".error \"TT_INSN in non-tensix code\"" ::"X"(unsigned(ENCODING))); })
#else
#define TTI_INSN(ENCODING) ({ __asm__ __volatile__(".error \"TTI_INSN in non-tensix code\"" ::"X"(unsigned(ENCODING))); })
#define TT_INSN(ENCODING)  ({ __asm__ __volatile__(".error \"TT_INSN in non-tensix code\"" ::"X"(unsigned(ENCODING))); })
#endif

#define TT_OP(opcode, params) ((opcode << 24) + params)

#define TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) \
    TT_OP(0x26, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_MVMUL_VALID(clear_dvalid, instr_mod19, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(addr_mode, 5) && ckernel::is_valid(dst, 14))
#define TT_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)  TT_INSN(TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst))
#define TTI_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) TTI_INSN(TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst))
