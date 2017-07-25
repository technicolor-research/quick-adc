//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef DISTANCES_HPP_
#define DISTANCES_HPP_

#include <immintrin.h>
extern "C" {
#include <cblas.h>
}
#include "quantizers.hpp"
#include "config.h"

const int SIMD_FLOATS = 8; // AVX: 256bits = 8 x 32bits

template<int DSQ>
const float* subv(const float* vec, int i) {
    return vec + (i * DSQ);
}

static inline float reduceadd(__m256 result) {
    const __m256 result_perm4 = _mm256_permute2f128_ps(result, result, 0x81);
    result = _mm256_add_ps(result, result_perm4);
    const __m256 result_perm2 = _mm256_permute_ps(result, 0xe);
    result = _mm256_add_ps(result, result_perm2);
    const __m256 result_perm1 = _mm256_permute_ps(result, 0x1);
    result = _mm256_add_ps(result, result_perm1);
    return _mm_cvtss_f32(_mm256_castps256_ps128(result));
}

template<int N>
float avxnorm(const float* vec_a, const float* vec_b) {
    __m256 result = _mm256_setzero_ps();
    for(int block_i = 0; block_i < N; ++block_i) {
        const __m256 block_a = _mm256_loadu_ps(vec_a + block_i*8);
        const __m256 block_b = _mm256_loadu_ps(vec_b + block_i*8);
        const __m256 diff = _mm256_sub_ps(block_a, block_b);
        const __m256 partial = _mm256_mul_ps(diff, diff);
        result = _mm256_add_ps(result, partial);
    }
    return reduceadd(result);
}

inline float norm_4(const float* vec) {
    float result = 0;
    for(int i = 0; i < 4; ++i) {
        result += vec[i] * vec[i];
    }
    return result;
}

#ifdef AVX2
template<int BLOCK_COUNT, int REMAINDER>
float fmanorm(const float* vec_a, const float* vec_b) {
    __m256 result = _mm256_setzero_ps();
    for(int block_i = 0; block_i < BLOCK_COUNT; ++block_i) {
        const __m256 block_a = _mm256_loadu_ps(vec_a + block_i*SIMD_FLOATS);
        const __m256 block_b = _mm256_loadu_ps(vec_b + block_i*SIMD_FLOATS);
        const __m256 diff = _mm256_sub_ps(block_a, block_b);
        result = _mm256_fmadd_ps(diff, diff, result);
    }
    float norm = reduceadd(result);
    vec_a += BLOCK_COUNT * SIMD_FLOATS;
    vec_b += BLOCK_COUNT * SIMD_FLOATS;
    for(int rem_i = 0; rem_i < REMAINDER; ++rem_i) {
        const float diff = vec_b[rem_i] - vec_a[rem_i];
        norm += diff * diff;
    }
    return norm;
}

template<int BLOCK_COUNT, int REMAINDER>
float fmanorm(const float* vec_a) {
    __m256 result = _mm256_setzero_ps();
    for(int block_i = 0; block_i < BLOCK_COUNT; ++block_i) {
        const __m256 block_a = _mm256_loadu_ps(vec_a + block_i*SIMD_FLOATS);
        result = _mm256_fmadd_ps(block_a, block_a, result);
    }
    float norm = reduceadd(result);
    vec_a += BLOCK_COUNT * SIMD_FLOATS;
    for(int rem_i = 0; rem_i < REMAINDER; ++rem_i) {
        norm += vec_a[rem_i] * vec_a[rem_i];
    }
    return norm;
}

#else
template<int N>
float fmanorm(const float* vec_a, const float* vec_b) {
    __m256 result = _mm256_setzero_ps();
    for(int block_i = 0; block_i < N; ++block_i) {
        const __m256 block_a = _mm256_loadu_ps(vec_a + block_i*8);
        const __m256 block_b = _mm256_loadu_ps(vec_b + block_i*8);
        const __m256 diff = _mm256_sub_ps(block_a, block_b);
        const __m256 partial = _mm256_mul_ps(diff, diff);
        result = _mm256_add_ps(result, partial);
    }
    return reduceadd(result);
}

template<int N>
float fmanorm(const float* vec_a) {
    __m256 result = _mm256_setzero_ps();
    for(int block_i = 0; block_i < N; ++block_i) {
        const __m256 block_a = _mm256_loadu_ps(vec_a + block_i*8);
        const __m256 partial = _mm256_mul_ps(block_a, block_a);
        result = _mm256_add_ps(result, partial);
    }
    return reduceadd(result);
}
#endif

template<int DSQ, decltype(avxnorm<4>)* normfunc>
void compute_dists_single_simd(float* dists, const base_pq& pq,
        const float* vector) {
    const int cent_count = pq.sq_centroid_count();
    for (int sq_i = 0; sq_i < pq.sq_count; ++sq_i) {
        const float* subvector = subv<DSQ>(vector, sq_i);
        for (int cent_i = 0; cent_i < cent_count; ++cent_i) {
            const float* centroid = subv<DSQ>(pq.centroids[sq_i], cent_i);
            dists[sq_i * cent_count + cent_i] = normfunc(subvector, centroid);
        }
    }
}

template<int DSQ>
void compute_dists_single_simd(float* dists, const base_pq& pq,
        const float* vector) {

    const int SIMD_BLOCKS = DSQ / SIMD_FLOATS;
    const int REMAINDER = DSQ % SIMD_FLOATS;

    const int cent_count = pq.sq_centroid_count();
    for (int sq_i = 0; sq_i < pq.sq_count; ++sq_i) {
        const float* subvector = subv<DSQ>(vector, sq_i);
        for (int cent_i = 0; cent_i < cent_count; ++cent_i) {
            const float* centroid = subv<DSQ>(pq.centroids[sq_i], cent_i);
            dists[sq_i * cent_count + cent_i] = fmanorm<SIMD_BLOCKS, REMAINDER>(
                    subvector, centroid);
        }
    }
}

template<int DSQ>
void compute_cross_dists_blas(float* dists, const float* centroids, int cent_count,
        const float* vectors, int vec_count, int dists_dim) {

    const int SIMD_BLOCKS = DSQ / SIMD_FLOATS;
    const int REMAINDER = DSQ % SIMD_FLOATS;
    //const int REMAINDER = 0;

    // Centroids norms
    std::unique_ptr<float[]> centroid_norms = std::make_unique<float[]>(cent_count);
    for (int cent_i = 0; cent_i < cent_count; cent_i++) {
        const float* cent = centroids + DSQ * cent_i;
        centroid_norms[cent_i] = fmanorm<SIMD_BLOCKS, REMAINDER>(cent);
    }

    // Distance matrix
    float* dists_line = dists;
    const float* vec = vectors;
    for (int vec_i = 0; vec_i < vec_count; ++vec_i) {
        const float vec_norm = fmanorm<SIMD_BLOCKS, REMAINDER>(vec);
        for (int cent_i = 0; cent_i < cent_count; ++cent_i) {
            dists_line[cent_i] = vec_norm + centroid_norms[cent_i];
        }
        vec += DSQ;
        dists_line += dists_dim;
    }

    // BLAS Call
    const float alpha = -2;
    const float beta = 1;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, vec_count, cent_count,
            DSQ, alpha, vectors, DSQ, centroids, DSQ, beta, dists, dists_dim);
}

template<>
inline void compute_cross_dists_blas<4>(float* dists, const float* centroids, int cent_count,
        const float* vectors, int vec_count, int dists_dim) {

    const int DSQ = 4;

    // Centroids norms
    std::unique_ptr<float[]> centroid_norms = std::make_unique<float[]>(cent_count);
    for (int cent_i = 0; cent_i < cent_count; cent_i++) {
        const float* cent = centroids + DSQ * cent_i;
        centroid_norms[cent_i] = norm_4(cent);
    }

    // Distance matrix
    float* dists_line = dists;
    const float* vec = vectors;
    for (int vec_i = 0; vec_i < vec_count; ++vec_i) {
        const float vec_norm = norm_4(vec);
        for (int cent_i = 0; cent_i < cent_count; ++cent_i) {
            dists_line[cent_i] = vec_norm + centroid_norms[cent_i];
        }
        vec += DSQ;
        dists_line += dists_dim;
    }

    // BLAS Call
    const float alpha = -2;
    const float beta = 1;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, vec_count, cent_count,
            DSQ, alpha, vectors, DSQ, centroids, DSQ, beta, dists, dists_dim);
}



template<int DSQ>
void compute_dists_multiple_blas(float* dists, const base_pq& pq,
        const float* vectors, const int count) {

    // Subvectors
    std::unique_ptr<float[]> subvectors(new float[count * DSQ]);

    const int sq_cent_count = pq.sq_centroid_count();
    const int table_dim = pq.sq_count * sq_cent_count;
    for (int sq_i = 0; sq_i < pq.sq_count; ++sq_i) {
        extract_subvectors(vectors, pq.dim, count, DSQ, sq_i, subvectors.get());
        compute_cross_dists_blas<DSQ>(dists + sq_i * sq_cent_count,
                pq.centroids[sq_i], sq_cent_count, subvectors.get(), count,
                table_dim);
    }

}

struct centroids_getter {

    virtual float** centroids() const = 0;
    virtual int sq_centroid_count() const = 0;
    virtual int sq_count() const = 0;
    virtual int dim() const = 0;
    virtual int sq_dim() const = 0;
    virtual ~centroids_getter() {
    }

};


struct base_centroids_getter : centroids_getter {
    base_pq* pq;

    base_centroids_getter(base_pq* pq_): pq(pq_) {};

    float** centroids() const {
        return pq->centroids.get();
    }

    int sq_centroid_count() const {
        return pq->sq_centroid_count();
    }

    int sq_count() const {
        return pq->sq_count;
    }

    int dim() const {
        return pq->dim;
    }

    int sq_dim() const {
        return pq->sq_dim();
    }

};

template<int DSQ>
void compute_dists_multiple_blas_cg(float* dists, const centroids_getter& cg,
        const float* vectors, const int count) {

    // Subvectors
    std::unique_ptr<float[]> subvectors(new float[count * DSQ]);

    const int sq_cent_count = cg.sq_centroid_count();
    const int table_dim = cg.sq_count() * sq_cent_count;
    for (int sq_i = 0; sq_i < cg.sq_count(); ++sq_i) {
        extract_subvectors(vectors, cg.dim(), count, DSQ, sq_i, subvectors.get());
        compute_cross_dists_blas<DSQ>(dists + sq_i * sq_cent_count,
                cg.centroids()[sq_i], sq_cent_count, subvectors.get(), count,
                table_dim);
    }
}

template<int DSQ>
void compute_dists_single_simd_cg(float* dists, const centroids_getter& cg,
        const float* vector) {

    const int SIMD_BLOCKS = DSQ / SIMD_FLOATS;
    const int REMAINDER = DSQ % SIMD_FLOATS;

    const int cent_count = cg.sq_centroid_count();
    for (int sq_i = 0; sq_i < cg.sq_count(); ++sq_i) {
        const float* subvector = subv<DSQ>(vector, sq_i);
        const float* sq_centroids = cg.centroids()[sq_i];
        for (int cent_i = 0; cent_i < cent_count; ++cent_i) {
            const float* centroid = subv<DSQ>(sq_centroids, cent_i);
            dists[sq_i * cent_count + cent_i] = fmanorm<SIMD_BLOCKS, REMAINDER>(
                    subvector, centroid);
        }
    }
}

typedef decltype(&compute_dists_multiple_blas_cg<128>) dists_mutiple_func;
dists_mutiple_func get_dists_mutiple_function(int sq_dim);

typedef decltype(&compute_dists_single_simd_cg<128>) dists_func;
dists_func get_dists_function(int sq_dim);

typedef decltype(&compute_cross_dists_blas<128>) cross_dists_func;
cross_dists_func get_cross_dists_func(int dim);

#endif
