//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef SIMD_LAYOUT_HPP_
#define SIMD_LAYOUT_HPP_

#include <cstdint>

struct source_partition {
    const std::uint8_t* data;
    int code_size;
    unsigned code_count;

    void shift(int shift_count) {
        code_count -= shift_count;
        data += shift_count * code_size;
    }

    const std::uint8_t* get_code(long code_i) const {
        return data + code_i * code_size;
    }
};

inline int compute_vector_block_count(const unsigned code_count, const int simd_size) {
    return (code_count + simd_size - 1) / simd_size;
}

inline long compute_interleaved_size_4(const unsigned code_count,
        const int code_size, const int simd_size) {
    return compute_vector_block_count(code_count, simd_size)
            * static_cast<long>(code_size) * simd_size;
}

void interleave_simd_block_4(std::uint8_t* dst, const source_partition& src,
        int simd_size, int byte_i) {
    unsigned simd_size_unsigned = static_cast<unsigned>(simd_size);
    for (unsigned simd_i = 0; simd_i < simd_size_unsigned; ++simd_i) {
        const std::uint8_t* code;
        if (simd_i < src.code_count) {
            code = src.get_code(simd_i);
        } else {
            code = src.get_code(src.code_count - 1);
        }
        dst[simd_i] = code[byte_i];
    }
}

void interleave_partition_4(std::uint8_t* dst, source_partition& src, int simd_size) {
    const int block_count = compute_vector_block_count(src.code_count,
            simd_size);
    for (int block_i = 0; block_i < block_count; ++block_i) {
        for(int byte_i = 0; byte_i < src.code_size; ++byte_i) {
            interleave_simd_block_4(dst, src, simd_size, byte_i);
            dst += simd_size;
        }
        src.shift(simd_size);
    }
}

#endif /* SIMD_LAYOUT_HPP_ */
