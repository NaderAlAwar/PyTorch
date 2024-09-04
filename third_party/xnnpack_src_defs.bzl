"""
Auto-generated by generate-wrappers.py script. Do not modify
"""

PROD_ARMSIMD32_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/armsimd32.c",
]

PROD_NEONFP16ARITH_AARCH64_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neonfp16arith-aarch64.c",
]

AARCH64_ASM_MICROKERNEL_SRCS = [
    "XNNPACK/src/f16-gemm/gen/f16-gemm-1x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-4x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemm-8x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-1x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-4x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-6x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-gemm/gen/f16-gemminc-8x8-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-igemm/f16-igemm-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-igemm/f16-igemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S",
    "XNNPACK/src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S",
    "XNNPACK/src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S",
    "XNNPACK/src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S",
    "XNNPACK/src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S",
    "XNNPACK/src/f32-dwconv/f32-dwconv-9p4c-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-dwconv/f32-dwconv-9p4c-minmax-asm-aarch64-neonfma.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a73.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-goi-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-goi-1x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-goi-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-1x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-1x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-4x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-cortex-a73.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-gemminc/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-igemm/f32-igemm-1x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-igemm/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-igemm/f32-igemm-4x12-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-igemm/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a55.S",
    "XNNPACK/src/f32-igemm/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a73.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S",
    "XNNPACK/src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S",
    "XNNPACK/src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondotfp16arith-cortex-a55.S",
    "XNNPACK/src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S",
    "XNNPACK/src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S",
    "XNNPACK/src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld64.S",
    "XNNPACK/src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S",
    "XNNPACK/src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mull.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S",
]

PROD_AVXVNNI_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avxvnni.c",
]

SUBGRAPH_SRCS = [
    "XNNPACK/src/memory-planner.c",
    "XNNPACK/src/runtime.c",
    "XNNPACK/src/subgraph.c",
    "XNNPACK/src/subgraph/abs.c",
    "XNNPACK/src/subgraph/add2.c",
    "XNNPACK/src/subgraph/argmax-pooling-2d.c",
    "XNNPACK/src/subgraph/average-pooling-2d.c",
    "XNNPACK/src/subgraph/bankers-rounding.c",
    "XNNPACK/src/subgraph/batch-matrix-multiply.c",
    "XNNPACK/src/subgraph/ceiling.c",
    "XNNPACK/src/subgraph/clamp.c",
    "XNNPACK/src/subgraph/concatenate.c",
    "XNNPACK/src/subgraph/convert.c",
    "XNNPACK/src/subgraph/convolution-2d.c",
    "XNNPACK/src/subgraph/copy.c",
    "XNNPACK/src/subgraph/copysign.c",
    "XNNPACK/src/subgraph/deconvolution-2d.c",
    "XNNPACK/src/subgraph/depth-to-space-2d.c",
    "XNNPACK/src/subgraph/depthwise-convolution-2d.c",
    "XNNPACK/src/subgraph/divide.c",
    "XNNPACK/src/subgraph/elu.c",
    "XNNPACK/src/subgraph/even-split.c",
    "XNNPACK/src/subgraph/exp.c",
    "XNNPACK/src/subgraph/floor.c",
    "XNNPACK/src/subgraph/fully-connected-sparse.c",
    "XNNPACK/src/subgraph/fully-connected.c",
    "XNNPACK/src/subgraph/gelu.c",
    "XNNPACK/src/subgraph/global-average-pooling.c",
    "XNNPACK/src/subgraph/global-sum-pooling.c",
    "XNNPACK/src/subgraph/hardswish.c",
    "XNNPACK/src/subgraph/leaky-relu.c",
    "XNNPACK/src/subgraph/log.c",
    "XNNPACK/src/subgraph/max-pooling-2d.c",
    "XNNPACK/src/subgraph/maximum2.c",
    "XNNPACK/src/subgraph/minimum2.c",
    "XNNPACK/src/subgraph/multiply2.c",
    "XNNPACK/src/subgraph/negate.c",
    "XNNPACK/src/subgraph/prelu.c",
    "XNNPACK/src/subgraph/reciprocal-square-root.c",
    "XNNPACK/src/subgraph/reshape-helpers.c",
    "XNNPACK/src/subgraph/scaled-dot-product-attention.c",
    "XNNPACK/src/subgraph/sigmoid.c",
    "XNNPACK/src/subgraph/softmax.c",
    "XNNPACK/src/subgraph/space-to-depth-2d.c",
    "XNNPACK/src/subgraph/square-root.c",
    "XNNPACK/src/subgraph/square.c",
    "XNNPACK/src/subgraph/squared-difference.c",
    "XNNPACK/src/subgraph/static-constant-pad.c",
    "XNNPACK/src/subgraph/static-mean.c",
    "XNNPACK/src/subgraph/static-reshape.c",
    "XNNPACK/src/subgraph/static-resize-bilinear-2d.c",
    "XNNPACK/src/subgraph/static-slice.c",
    "XNNPACK/src/subgraph/static-transpose.c",
    "XNNPACK/src/subgraph/subtract.c",
    "XNNPACK/src/subgraph/tanh.c",
    "XNNPACK/src/subgraph/unpooling-2d.c",
    "XNNPACK/src/subgraph/validation.c",
    "XNNPACK/src/tensor.c",
]

PROD_AVX512VNNIGFNI_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx512vnnigfni.c",
]

PROD_AVX512VNNI_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx512vnni.c",
]

PROD_SSE2_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/sse2.c",
]

PROD_NEONDOT_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neondot.c",
]

PROD_SSE41_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/sse41.c",
]

PROD_SSE_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/sse.c",
]

PROD_NEONFP16ARITH_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neonfp16arith.c",
]

PROD_NEONV8_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neonv8.c",
]

PROD_NEONFP16_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neonfp16.c",
]

XNNPACK_SRCS = [
    "XNNPACK/src/configs/argmaxpool-config.c",
    "XNNPACK/src/configs/avgpool-config.c",
    "XNNPACK/src/configs/binary-elementwise-config.c",
    "XNNPACK/src/configs/cmul-config.c",
    "XNNPACK/src/configs/conv-hwc2chw-config.c",
    "XNNPACK/src/configs/dwconv-config.c",
    "XNNPACK/src/configs/dwconv2d-chw-config.c",
    "XNNPACK/src/configs/experiments-config.c",
    "XNNPACK/src/configs/gavgpool-config.c",
    "XNNPACK/src/configs/gavgpool-cw-config.c",
    "XNNPACK/src/configs/gemm-config.c",
    "XNNPACK/src/configs/ibilinear-chw-config.c",
    "XNNPACK/src/configs/ibilinear-config.c",
    "XNNPACK/src/configs/lut32norm-config.c",
    "XNNPACK/src/configs/maxpool-config.c",
    "XNNPACK/src/configs/pavgpool-config.c",
    "XNNPACK/src/configs/prelu-config.c",
    "XNNPACK/src/configs/raddstoreexpminusmax-config.c",
    "XNNPACK/src/configs/reduce-config.c",
    "XNNPACK/src/configs/rmax-config.c",
    "XNNPACK/src/configs/spmm-config.c",
    "XNNPACK/src/configs/transpose-config.c",
    "XNNPACK/src/configs/unary-elementwise-config.c",
    "XNNPACK/src/configs/unpool-config.c",
    "XNNPACK/src/configs/vmulcaddc-config.c",
    "XNNPACK/src/configs/xx-fill-config.c",
    "XNNPACK/src/configs/xx-pad-config.c",
    "XNNPACK/src/configs/x8-lut-config.c",
    "XNNPACK/src/configs/zip-config.c",
    "XNNPACK/src/init.c",
    "XNNPACK/src/params.c",
]

PROD_AVX_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx.c",
]

PROD_AVX512SKX_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx512skx.c",
]

PROD_NEONDOTFP16ARITH_AARCH64_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neondotfp16-aarch64.c",
]

PROD_FP16ARITH_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/fp16arith.c",
]

PROD_FMA_MICROKERNEL_SRCS = [
]

OPERATOR_SRCS = [
    "XNNPACK/src/operator-delete.c",
    "XNNPACK/src/operators/argmax-pooling-nhwc.c",
    "XNNPACK/src/operators/average-pooling-nhwc.c",
    "XNNPACK/src/operators/batch-matrix-multiply-nc.c",
    "XNNPACK/src/operators/binary-elementwise-nd.c",
    "XNNPACK/src/operators/channel-shuffle-nc.c",
    "XNNPACK/src/operators/constant-pad-nd.c",
    "XNNPACK/src/operators/convolution-nchw.c",
    "XNNPACK/src/operators/convolution-nhwc.c",
    "XNNPACK/src/operators/deconvolution-nhwc.c",
    "XNNPACK/src/operators/dynamic-fully-connected-nc.c",
    "XNNPACK/src/operators/fully-connected-nc.c",
    "XNNPACK/src/operators/global-average-pooling-ncw.c",
    "XNNPACK/src/operators/global-average-pooling-nwc.c",
    "XNNPACK/src/operators/lut-elementwise-nc.c",
    "XNNPACK/src/operators/max-pooling-nhwc.c",
    "XNNPACK/src/operators/prelu-nc.c",
    "XNNPACK/src/operators/reduce-nd.c",
    "XNNPACK/src/operators/resize-bilinear-nchw.c",
    "XNNPACK/src/operators/resize-bilinear-nhwc.c",
    "XNNPACK/src/operators/rope-nthc.c",
    "XNNPACK/src/operators/scaled-dot-product-attention-nhtc.c",
    "XNNPACK/src/operators/slice-nd.c",
    "XNNPACK/src/operators/softmax-nc.c",
    "XNNPACK/src/operators/transpose-nd.c",
    "XNNPACK/src/operators/unary-elementwise-nc.c",
    "XNNPACK/src/operators/unpooling-nhwc.c",
]

PROD_NEONI8MM_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neoni8mm.c",
]

PROD_AVX512F_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx512f.c",
]

JIT_SRCS = [
]

PROD_F16C_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/f16c.c",
]

PROD_NEON_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neon.c",
]

PROD_SCALAR_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/scalar.c",
]

PROD_NEONDOT_AARCH64_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neondot-aarch64.c",
]

PROD_FMA3_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/fma3.c",
]

LOGGING_SRCS = [
    "XNNPACK/src/enums/allocation-type.c",
    "XNNPACK/src/enums/datatype-strings.c",
    "XNNPACK/src/enums/microkernel-type.c",
    "XNNPACK/src/enums/node-type.c",
    "XNNPACK/src/enums/operator-type.c",
    "XNNPACK/src/log.c",
]

PROD_NEONFMA_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neonfma.c",
]

PROD_AVX2_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx2.c",
]

PROD_AVX512VBMI_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/avx512vbmi.c",
]

PROD_RVV_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/rvv.c",
]

PROD_NEONDOTFP16ARITH_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neondotfp16arith.c",
]

PROD_XOP_MICROKERNEL_SRCS = [
]

AARCH32_ASM_MICROKERNEL_SRCS = [
    "XNNPACK/src/cs16-bfly4/cs16-bfly4-samples1-asm-aarch32-neon-x1.S",
    "XNNPACK/src/cs16-bfly4/cs16-bfly4-samples1-asm-aarch32-neon-x2.S",
    "XNNPACK/src/cs16-bfly4/cs16-bfly4-samples1-asm-aarch32-neon-x4.S",
    "XNNPACK/src/cs16-fftr/cs16-fftr-asm-aarch32-neon-x1.S",
    "XNNPACK/src/cs16-fftr/cs16-fftr-asm-aarch32-neon-x4.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch32-neon-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x4-asm-aarch32-vfp-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x4-minmax-asm-aarch32-vfp-ld64.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a7.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a53.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a55.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a75-prfm.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a75.S",
    "XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-ld64.S",
    "XNNPACK/src/f32-igemm/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a55.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch32-neon-cortex-a53.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a7.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a53.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a75-prfm.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a75.S",
    "XNNPACK/src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-ld64.S",
    "XNNPACK/src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondotfp16arith-cortex-a55.S",
    "XNNPACK/src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c4-minmax-asm-aarch32-neondotfp16arith-cortex-a55.S",
    "XNNPACK/src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondot-cortex-a55.S",
    "XNNPACK/src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c4-minmax-asm-aarch32-neondot-cortex-a55.S",
    "XNNPACK/src/qs8-qc8w-dwconv/qs8-qc8w-dwconv-3p8c-minmax-fp32-asm-aarch32-neonv8-mla8-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-dwconv/qs8-qc8w-dwconv-3p16c-minmax-fp32-asm-aarch32-neonv8-mla8-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-cortex-a55.S",
    "XNNPACK/src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-ld64.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c4-minmax-fp32-asm-aarch32-neondot-cortex-a55.S",
    "XNNPACK/src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c4-minmax-fp32-asm-aarch32-neondot-ld64.S",
    "XNNPACK/src/qs16-qs8-vcvt/qs16-qs8-vcvt-asm-aarch32-neon-u16.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S",
    "XNNPACK/src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S",
    "XNNPACK/src/u32-filterbank-accumulate/u32-filterbank-accumulate-asm-aarch32-arm-x1.S",
    "XNNPACK/src/u32-filterbank-accumulate/u32-filterbank-accumulate-asm-aarch32-neon-x1.S",
    "XNNPACK/src/u32-filterbank-accumulate/u32-filterbank-accumulate-asm-aarch32-neon-x2.S",
]

PROD_SSSE3_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/ssse3.c",
]

TABLE_SRCS = [
    "XNNPACK/src/tables/exp2-k-over-64.c",
    "XNNPACK/src/tables/exp2-k-over-2048.c",
    "XNNPACK/src/tables/exp2minus-k-over-4.c",
    "XNNPACK/src/tables/exp2minus-k-over-8.c",
    "XNNPACK/src/tables/exp2minus-k-over-16.c",
    "XNNPACK/src/tables/exp2minus-k-over-32.c",
    "XNNPACK/src/tables/exp2minus-k-over-64.c",
    "XNNPACK/src/tables/exp2minus-k-over-2048.c",
    "XNNPACK/src/tables/vlog.c",
]

PROD_NEON_AARCH64_MICROKERNEL_SRCS = [
    "XNNPACK/src/amalgam/gen/neon-aarch64.c",
    "XNNPACK/src/amalgam/gen/neonfma-aarch64.c",
]
