//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include <memory>
#include "binheap.hpp"
#include "distances.hpp"

const int BLOCK_VECS = 256;
const int BLOCK_NEIGHS = 256;

static inline void add_candidates_heaps(float* dist_block, kv_binheap<int, float>* heaps,
                          int block_count_vec, int block_count_neigh, int base_neigh) {
    float* dist_line = dist_block;
    for(int vec_i = 0; vec_i < block_count_vec; ++vec_i) {
        kv_binheap<int, float>& h = heaps[vec_i];
        for(int neigh_i = 0; neigh_i < block_count_neigh; ++neigh_i) {
            h.push(base_neigh + neigh_i, dist_line[neigh_i]);
        }
        dist_line += block_count_neigh;
    }
}

void find_k_neighbors(
        const int vector_count, const int neighbor_count, const int dim, const int k,
        const float* vectors, const float* neighbors, int* assignements) {

    std::unique_ptr<kv_binheap<int, float>[]> heaps = std::unique_ptr<kv_binheap<int, float>[]>(
                new kv_binheap<int, float>[BLOCK_VECS]);
    for(int h_i = 0; h_i < BLOCK_VECS; ++h_i) {
        heaps[h_i].reset_capacity(k);
    }

    std::unique_ptr<float[]> dists_block(new float[BLOCK_VECS * BLOCK_NEIGHS]);
    std::unique_ptr<float[]> sorted_distances(new float[k]);
    cross_dists_func dists_func = get_cross_dists_func(dim);

    // Shifted data structures
    int* shifted_assignements = assignements;
    const float* shifted_vectors = vectors;

    for(int vec_i = 0; vec_i < vector_count; vec_i += BLOCK_VECS) {
        int block_count_vec = std::min(BLOCK_VECS, vector_count - vec_i);

        // Reset heaps
        for(int v_i = 0; v_i < block_count_vec; ++v_i) {
            heaps[v_i].reset();
        }

        // Compute distances and insert into heaps
        const float* shifted_neighbors = neighbors;
        for(int neigh_i = 0; neigh_i < neighbor_count; neigh_i += BLOCK_NEIGHS) {
            int block_count_neigh = std::min(BLOCK_NEIGHS, neighbor_count - neigh_i);
            dists_func(dists_block.get(), shifted_neighbors, block_count_neigh, shifted_vectors,
                       block_count_vec, block_count_neigh);
            add_candidates_heaps(dists_block.get(), heaps.get(), block_count_vec, block_count_neigh,
                                neigh_i);
            shifted_neighbors += block_count_neigh;
        }

        // Sort heaps
        for(int v_i = 0; v_i < block_count_vec; ++v_i) {
            heaps[v_i].sort(shifted_assignements, sorted_distances.get());
            shifted_assignements += k;
        }

        // Shift data structures
        shifted_vectors += block_count_vec * dim;
    }
}
