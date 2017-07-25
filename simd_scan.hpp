//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef SIMD_SCAN_HPP_
#define SIMD_SCAN_HPP_

#include <cstdint>
#include <cassert>
#include <immintrin.h>
#include <x86intrin.h>
#include "config.h"

const std::uint64_t masktable[] = { 0x0, 0x0, 0x1, 0x100, 0x2, 0x200, 0x201,
        0x20100, 0x3, 0x300, 0x301, 0x30100, 0x302, 0x30200, 0x30201, 0x3020100,
        0x4, 0x400, 0x401, 0x40100, 0x402, 0x40200, 0x40201, 0x4020100, 0x403,
        0x40300, 0x40301, 0x4030100, 0x40302, 0x4030200, 0x4030201, 0x403020100,
        0x5, 0x500, 0x501, 0x50100, 0x502, 0x50200, 0x50201, 0x5020100, 0x503,
        0x50300, 0x50301, 0x5030100, 0x50302, 0x5030200, 0x5030201, 0x503020100,
        0x504, 0x50400, 0x50401, 0x5040100, 0x50402, 0x5040200, 0x5040201,
        0x504020100, 0x50403, 0x5040300, 0x5040301, 0x504030100, 0x5040302,
        0x504030200, 0x504030201, 0x50403020100, 0x6, 0x600, 0x601, 0x60100,
        0x602, 0x60200, 0x60201, 0x6020100, 0x603, 0x60300, 0x60301, 0x6030100,
        0x60302, 0x6030200, 0x6030201, 0x603020100, 0x604, 0x60400, 0x60401,
        0x6040100, 0x60402, 0x6040200, 0x6040201, 0x604020100, 0x60403,
        0x6040300, 0x6040301, 0x604030100, 0x6040302, 0x604030200, 0x604030201,
        0x60403020100, 0x605, 0x60500, 0x60501, 0x6050100, 0x60502, 0x6050200,
        0x6050201, 0x605020100, 0x60503, 0x6050300, 0x6050301, 0x605030100,
        0x6050302, 0x605030200, 0x605030201, 0x60503020100, 0x60504, 0x6050400,
        0x6050401, 0x605040100, 0x6050402, 0x605040200, 0x605040201,
        0x60504020100, 0x6050403, 0x605040300, 0x605040301, 0x60504030100,
        0x605040302, 0x60504030200, 0x60504030201, 0x6050403020100, 0x7, 0x700,
        0x701, 0x70100, 0x702, 0x70200, 0x70201, 0x7020100, 0x703, 0x70300,
        0x70301, 0x7030100, 0x70302, 0x7030200, 0x7030201, 0x703020100, 0x704,
        0x70400, 0x70401, 0x7040100, 0x70402, 0x7040200, 0x7040201, 0x704020100,
        0x70403, 0x7040300, 0x7040301, 0x704030100, 0x7040302, 0x704030200,
        0x704030201, 0x70403020100, 0x705, 0x70500, 0x70501, 0x7050100, 0x70502,
        0x7050200, 0x7050201, 0x705020100, 0x70503, 0x7050300, 0x7050301,
        0x705030100, 0x7050302, 0x705030200, 0x705030201, 0x70503020100,
        0x70504, 0x7050400, 0x7050401, 0x705040100, 0x7050402, 0x705040200,
        0x705040201, 0x70504020100, 0x7050403, 0x705040300, 0x705040301,
        0x70504030100, 0x705040302, 0x70504030200, 0x70504030201,
        0x7050403020100, 0x706, 0x70600, 0x70601, 0x7060100, 0x70602, 0x7060200,
        0x7060201, 0x706020100, 0x70603, 0x7060300, 0x7060301, 0x706030100,
        0x7060302, 0x706030200, 0x706030201, 0x70603020100, 0x70604, 0x7060400,
        0x7060401, 0x706040100, 0x7060402, 0x706040200, 0x706040201,
        0x70604020100, 0x7060403, 0x706040300, 0x706040301, 0x70604030100,
        0x706040302, 0x70604030200, 0x70604030201, 0x7060403020100, 0x70605,
        0x7060500, 0x7060501, 0x706050100, 0x7060502, 0x706050200, 0x706050201,
        0x70605020100, 0x7060503, 0x706050300, 0x706050301, 0x70605030100,
        0x706050302, 0x70605030200, 0x70605030201, 0x7060503020100, 0x7060504,
        0x706050400, 0x706050401, 0x70605040100, 0x706050402, 0x70605040200,
        0x70605040201, 0x7060504020100, 0x706050403, 0x70605040300,
        0x70605040301, 0x7060504030100, 0x70605040302, 0x7060504030200,
        0x7060504030201, 0x706050403020100 };

template<typename T>
FORCE_INLINE
inline void bh_push(kv_binheap<unsigned, T>& bh, const unsigned* labels,
        unsigned code_i, unsigned max_code_i, T cand) {
    code_i = std::min(code_i, max_code_i);
    if(labels == nullptr) {
        bh.push(code_i, cand);
    } else {
        bh.push(labels[code_i], cand);
    }
}

FORCE_INLINE
inline void compare_extract_matches_sse(const __m128i& __restrict__ candidates, __m128i& __restrict__ bh_bound_sse,
        const unsigned scanned, const unsigned max_scan,
        std::int8_t (&candidates_mem)[16],
        kv_binheap<unsigned, std::int8_t>& bh, const unsigned* labels,
        const unsigned labels_offset = 0) {

    const __m128i compare = _mm_cmplt_epi8(candidates, bh_bound_sse);
    int cmp = _mm_movemask_epi8(compare);

    if(cmp) {
        const unsigned first_code_i = labels_offset + scanned;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(candidates_mem), candidates);

        // Check low quadword
        const std::uint8_t cmp_low = cmp & 0xff;
        if (cmp_low) {
            const int match_count = _popcnt32(cmp_low);
            std::uint64_t match_pos = masktable[cmp_low];

            for (int i = 0; i < match_count; ++i) {
                const std::uint8_t pos = match_pos & 0xff;
                match_pos >>= 8;
                bh_push(bh, labels, first_code_i + pos, max_scan,
                        candidates_mem[pos]);
            }
        }

        // Check high quadword
        const std::uint8_t cmp_high = (cmp >> 8);
        if (cmp_high) {
            const int match_count = _popcnt32(cmp_high);
            std::uint64_t match_pos = masktable[cmp_high] + 0x0808080808080808;

            for (int i = 0; i < match_count; ++i) {
                const std::uint8_t pos = match_pos & 0xff;
                match_pos >>= 8;
                bh_push(bh, labels, first_code_i + pos, max_scan,
                        candidates_mem[pos]);
            }
        }
        bh_bound_sse = _mm_set1_epi8(bh.max());
    }
}

static inline __m256i _mm256_set_m128i(__m128i high,  __m128i low){
    __m256i lowb = _mm256_castsi128_si256(low);
    return _mm256_insertf128_si256(lowb,high, 1);
}

template<int SQ_COUNT>
void scan_avx_4(const std::uint8_t* __restrict__ partition, const unsigned* labels,
        const unsigned labels_offset,
        const unsigned size, const __m128i qdists[],
        kv_binheap<unsigned, std::int8_t>& bh) {

    const int VECTORS_PER_BLOCK = 16;
    const __m256i low_mask_avx = _mm256_set1_epi8(0x0f);

    // Load distance tables
    const int ROW_COUNT = SQ_COUNT / 4;
    const int TABLE_COUNT = ROW_COUNT * 2;

    __m256i qdists_m256[TABLE_COUNT];
    for(int row_i = 0; row_i < ROW_COUNT; ++row_i) {
        qdists_m256[2*row_i] = _mm256_set_m128i(qdists[4*row_i+2], qdists[4*row_i]);
        qdists_m256[2*row_i+1] = _mm256_set_m128i(qdists[4*row_i+3], qdists[4*row_i+1]);
    }

    // Binheap extraction
    __m128i bh_bound_sse = _mm_set1_epi8(bh.max());
    std::int8_t candidates_mem[VECTORS_PER_BLOCK];

    // Iterate over partition
    const unsigned max_scan = size - 1;
    unsigned scanned = 0;
    const __m256i* __restrict__ part  = reinterpret_cast<const __m256i*>(partition);
    while(scanned <= max_scan) {
        // Compute distances for block
        // Row 0
        __m256i comps0 = _mm256_loadu_si256(part);
        __m256i masked0 = _mm256_and_si256(comps0, low_mask_avx);
        __m256i twolane_sum = _mm256_shuffle_epi8(qdists_m256[0], masked0);
        comps0 = _mm256_srli_epi64(comps0, 4);
        masked0 = _mm256_and_si256(comps0, low_mask_avx);
        __m256i partial = _mm256_shuffle_epi8(qdists_m256[1], masked0);
        twolane_sum = _mm256_adds_epi8(partial, twolane_sum);
        // Rows 1..ROW_COUNT
        for(int row_i = 1; row_i < ROW_COUNT; ++row_i) {
            // Lookup add (low)
            __m256i comps = _mm256_loadu_si256(part + row_i);
            __m256i masked = _mm256_and_si256(comps, low_mask_avx);
            __m256i partial = _mm256_shuffle_epi8(qdists_m256[2*row_i], masked);
            twolane_sum = _mm256_adds_epi8(partial, twolane_sum);
            // Lookup add (high)
            comps = _mm256_srli_epi64(comps, 4);
            masked = _mm256_and_si256(comps, low_mask_avx);
            partial = _mm256_shuffle_epi8(qdists_m256[2*row_i+1], masked);
            twolane_sum = _mm256_adds_epi8(partial, twolane_sum);
        }

        // Reduce
        const __m256i twolane_sum_permuted = _mm256_permute2f128_si256(twolane_sum, twolane_sum, 0x13);
        const  __m256i twolane_candidates = _mm256_adds_epi8(twolane_sum_permuted, twolane_sum);
        const __m128i candidates =  _mm256_castsi256_si128(twolane_candidates);

        // Compare
        compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
                candidates_mem, bh, labels, labels_offset);
        scanned += VECTORS_PER_BLOCK;
        part += ROW_COUNT;
    }
}

void scan_avx_16_4(const std::uint8_t* __restrict__ partition, const unsigned* labels,
        const unsigned labels_offset,
        const unsigned size, const __m128i qdists[],
        kv_binheap<unsigned, std::int8_t>& bh) {

    const int SIMD_SIZE = 16;
    const __m256i low_mask_avx = _mm256_set1_epi8(0x0f);

    // Get tables
    const __m256i qdists_0_2 = _mm256_set_m128i(qdists[2], qdists[0]);
    const __m256i qdists_1_3 = _mm256_set_m128i(qdists[3], qdists[1]);

    const __m256i qdists_4_6 = _mm256_set_m128i(qdists[6], qdists[4]);
    const __m256i qdists_5_7 = _mm256_set_m128i(qdists[7], qdists[5]);

    const __m256i qdists_8_10 = _mm256_set_m128i(qdists[10], qdists[8]);
    const __m256i qdists_9_11 = _mm256_set_m128i(qdists[11], qdists[9]);

    const __m256i qdists_12_14 = _mm256_set_m128i(qdists[14], qdists[12]);
    const __m256i qdists_13_15 = _mm256_set_m128i(qdists[15], qdists[13]);

    // Binheap extraction
    __m128i bh_bound_sse = _mm_set1_epi8(bh.max());
    std::int8_t candidates_mem[16];

    // Iterate over vectors
    const __m256i* __restrict__ part  = reinterpret_cast<const __m256i*>(partition);
    const unsigned max_scan = size - 1;
    unsigned scanned = 0;
    while(scanned <= max_scan) {
        __m256i comps = _mm256_loadu_si256(part);
        // SQ 0 and 2
        __m256i comp_low = _mm256_and_si256(comps, low_mask_avx);
        __m256i twolane_sum = _mm256_shuffle_epi8(qdists_0_2, comp_low);
        // SQ 1 and 3
        comps = _mm256_srli_epi64(comps, 4);
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        __m256i partial = _mm256_shuffle_epi8(qdists_1_3, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);

        comps = _mm256_loadu_si256(part + 1);
        // SQ 4 and 6
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_4_6, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);
        // SQ 5 and 7
        comps = _mm256_srli_epi64(comps, 4);
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_5_7, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);

        comps = _mm256_loadu_si256(part + 2);
        // SQ 8 and 10
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_8_10, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);
        // SQ 9 and 11
        comps = _mm256_srli_epi64(comps, 4);
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_9_11, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);

        comps = _mm256_loadu_si256(part + 3);
        // SQ 12 and 14
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_12_14, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);
        // SQ 13 and 15
        comps = _mm256_srli_epi64(comps, 4);
        comp_low = _mm256_and_si256(comps, low_mask_avx);
        partial = _mm256_shuffle_epi8(qdists_13_15, comp_low);
        twolane_sum = _mm256_adds_epi8(twolane_sum, partial);

        // Reduce
        const __m256i twolane_sum_permuted = _mm256_permute2f128_si256(twolane_sum, twolane_sum, 0x13);
        const  __m256i twolane_candidates = _mm256_adds_epi8(twolane_sum_permuted, twolane_sum);
        const __m128i candidates =  _mm256_castsi256_si128(twolane_candidates);

        // Compare
        compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
                candidates_mem, bh, labels, labels_offset);
        scanned += SIMD_SIZE;
        part += 4;
    }
}

const int SQ_COUNT=16;
void scan_sse_16_4(const std::uint8_t* partition, const unsigned* labels,
        const unsigned labels_offset,
        const unsigned size, const __m128i (&qdists)[SQ_COUNT],
        kv_binheap<unsigned, std::int8_t>& bh) {

    const int SQ_BLOCKS_COUNT = SQ_COUNT / 2;
    const int SIMD_SIZE = 16;

    const __m128i* part = reinterpret_cast<const __m128i*>(partition);

    // Masks
    const std::uint64_t low_mask = 0x0f0f0f0f0f0f0f0f;
    const __m128i low_mask_sse = _mm_set_epi64x(low_mask, low_mask);

    // State
    __m128i bh_bound_sse = _mm_set1_epi8(bh.max());
    unsigned scanned = 0;
    std::int8_t candidates_mem[16];

    // Iterate over all vectors
    const unsigned max_scan = size - 1;
    while(scanned < size) {

        // Subquantizer 0
        const __m128i comps_01 = _mm_loadu_si128(part);
        const __m128i comps_0 = _mm_and_si128(comps_01, low_mask_sse);
        __m128i candidates = _mm_shuffle_epi8(qdists[0], comps_0);

        // Subquantizer 1
        const __m128i comps_01_shift = _mm_srli_epi64(comps_01, 4);
        const __m128i comps_1 = _mm_and_si128(comps_01_shift, low_mask_sse);
        const __m128i partial = _mm_shuffle_epi8(qdists[1], comps_1);
        candidates = _mm_adds_epi8(candidates, partial);

        // Subquantizers 2..SQ_COUNT
        for(int sq_blk_i = 1; sq_blk_i < SQ_BLOCKS_COUNT; ++sq_blk_i) {
            const int sq_i = sq_blk_i * 2;
            const __m128i comps = _mm_loadu_si128(part + sq_blk_i);
            // Low comps
            const __m128i comps_low = _mm_and_si128(comps, low_mask_sse);
            const __m128i partial_low = _mm_shuffle_epi8(qdists[sq_i], comps_low);
            candidates = _mm_adds_epi8(candidates, partial_low);
            // High comps
            const __m128i comps_shift = _mm_srli_epi64(comps, 4);
            const __m128i comps_high = _mm_and_si128(comps_shift, low_mask_sse);
            const __m128i partial_high = _mm_shuffle_epi8(qdists[sq_i + 1],
                    comps_high);
            candidates = _mm_adds_epi8(candidates, partial_high);
        }

        // Compare
        compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
                candidates_mem, bh, labels, labels_offset);
        scanned += SIMD_SIZE;
        part += SQ_BLOCKS_COUNT;
    }
}

#endif /* SIMD_SCAN_HPP_ */
