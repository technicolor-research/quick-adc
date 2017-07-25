//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include <cstdlib>
#include <iostream>
#include "distances.hpp"

dists_mutiple_func get_dists_mutiple_function(int sq_dim) {
    switch(sq_dim) {
    case 4:
        return compute_dists_multiple_blas_cg<4>;
    case 8:
        return compute_dists_multiple_blas_cg<8>;
    case 16:
        return compute_dists_multiple_blas_cg<16>;
    case 30:
        return compute_dists_multiple_blas_cg<30>;
    case 32:
        return compute_dists_multiple_blas_cg<32>;
    case 48:
        return compute_dists_multiple_blas_cg<48>;
    case 60:
        return compute_dists_multiple_blas_cg<60>;
    case 64:
        return compute_dists_multiple_blas_cg<64>;
    case 96:
        return compute_dists_multiple_blas_cg<96>;
    case 120:
        return compute_dists_multiple_blas_cg<120>;
    case 128:
        return compute_dists_multiple_blas_cg<128>;
    case 192:
        return compute_dists_multiple_blas_cg<192>;
    case 240:
        return compute_dists_multiple_blas_cg<240>;
    case 256:
        return compute_dists_multiple_blas_cg<256>;
    default:
        std::cerr << "Unsupported dimension " << sq_dim << std::endl;
        std::exit(1);
    }
}

dists_func get_dists_function(int sq_dim) {
    switch(sq_dim) {
    case 4:
        return compute_dists_single_simd_cg<4>;
    case 8:
        return compute_dists_single_simd_cg<8>;
    case 16:
        return compute_dists_single_simd_cg<16>;
    case 30:
        return compute_dists_single_simd_cg<30>;
    case 32:
        return compute_dists_single_simd_cg<32>;
    case 48:
        return compute_dists_single_simd_cg<48>;
    case 60:
        return compute_dists_single_simd_cg<60>;
    case 64:
        return compute_dists_single_simd_cg<64>;
    case 96:
        return compute_dists_single_simd_cg<96>;
    case 120:
        return compute_dists_single_simd_cg<120>;
    case 128:
        return compute_dists_single_simd_cg<128>;
    case 192:
        return compute_dists_single_simd_cg<192>;
    case 240:
        return compute_dists_single_simd_cg<240>;
    case 256:
        return compute_dists_single_simd_cg<256>;
    default:
        std::cerr << "Unsupported dimension " << sq_dim << std::endl;
        std::exit(1);
    }
}

cross_dists_func get_cross_dists_func(int dim) {
    switch (dim) {
    case 4:
        return compute_cross_dists_blas<4>;
    case 8:
        return compute_cross_dists_blas<8>;
    case 16:
        return compute_cross_dists_blas<16>;
    case 30:
        return compute_cross_dists_blas<30>;
    case 32:
        return compute_cross_dists_blas<32>;
    case 48:
        return compute_cross_dists_blas<48>;
    case 60:
        return compute_cross_dists_blas<60>;
    case 64:
        return compute_cross_dists_blas<64>;
    case 96:
        return compute_cross_dists_blas<96>;
    case 120:
        return compute_cross_dists_blas<120>;
    case 128:
        return compute_cross_dists_blas<128>;
    case 192:
        return compute_cross_dists_blas<192>;
    case 240:
        return compute_cross_dists_blas<240>;
    case 256:
        return compute_cross_dists_blas<256>;
    default:
        std::cerr << "Unsupported dimension " << dim << std::endl;
        std::exit(1);
    }
}
