//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef DATABASES_HPP_
#define DATABASES_HPP_

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <omp.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include "binheap.hpp"
#include "neighbors.hpp"
#include "distances.hpp"
#include "quantizers.hpp"
#include "vector_io.hpp"

const unsigned MIN_VECTORS_PER_THREAD = 10000;
inline int optimal_thread_count(unsigned vector_count) {
    const int thread_count = static_cast<int>(std::min(
            static_cast<unsigned>(omp_get_max_threads()),
            vector_count / MIN_VECTORS_PER_THREAD));
    return thread_count;
}

struct base_db {
    std::unique_ptr<base_pq> pq;

    base_db() {};
    base_db(std::unique_ptr<base_pq>&& pq_) :
            pq(std::move(pq_)) {
    }

    // Function to assign queries
    virtual void assign_compute_residuals(const float* vector, int multiple_assign,
            int* assignements, float* residuals) = 0;

    virtual void assign_compute_residuals_mutiple(const float* vectors,
            const int count, const int multiple_assign, int* assignements,
            float* residuals) = 0;

    virtual int partition_count() const = 0;

    virtual void get_partition(int part_i, const std::uint8_t*& codes,
            unsigned*& labels, unsigned& size) const = 0;

    virtual void free_partition(int part_i) = 0;

    virtual void add_vectors(float* vectors, unsigned count,
            unsigned labels_offset, int thread_count = 1) = 0;

    virtual void print(std::ostream& os) const = 0;

    virtual ~base_db() {};
};

inline void compute_thread_task(unsigned count, unsigned& off, unsigned& cnt) {
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
    const unsigned chunk_size = count / thread_count;

    off = thread_id * chunk_size;
    cnt = chunk_size;
    if (thread_id == thread_count - 1) {
        cnt = count - off;
    }
}

struct flat_db: public base_db {
    std::vector<std::uint8_t> codes;
    unsigned codes_count;

    flat_db() {};

    flat_db(std::unique_ptr<base_pq>&& pq_) :
        base_db(std::move(pq_)), codes_count(0) {

    }

    virtual void print(std::ostream& os) const {
        os << "Flat DB" << std::endl;
        os << *pq;
    }

    virtual void assign_compute_residuals(const float* vector, int multiple_assign,
            int* assignements, float* residuals) {
        const int dim = pq->dim;
        for(int ass_i = 0; ass_i < multiple_assign; ++ass_i) {
            assignements[ass_i] = 0;
            std::copy(vector, vector + dim, residuals);
            residuals += dim;
        }
    }

    virtual void assign_compute_residuals_mutiple(const float* vectors,
            const int count, const int multiple_assign, int* assignements,
            float* residuals) {
        const int dim = pq->dim;
        for (int vec_i = 0; vec_i < count; ++vec_i) {
            for (int ass_i = 0; ass_i < multiple_assign; ++ass_i) {
                assignements[ass_i] = 0;
                std::copy(vectors, vectors + dim, residuals);
                residuals += dim;
            }
            vectors += dim;
            assignements += multiple_assign;
        }
    }

    virtual int partition_count() const {
        return 1;
    }

    virtual void get_partition(int part_i, const std::uint8_t*& codes_,
            unsigned*& labels, unsigned& codes_count_) const {
        assert(part_i == 0);
        codes_ = codes.data();
        codes_count_ = codes_count;
        labels = nullptr;
    }

    virtual void free_partition(int part_i) {
        codes.resize(0);
        codes.shrink_to_fit();
        codes_count = 0;
    }

    virtual void add_vectors(float* vectors, unsigned count,
            unsigned labels_offset, int thread_count = 1) {

        const long code_size = pq->code_size();
        // Resize buffer
        if (labels_offset + count > codes_count) {
            codes_count = labels_offset + count;
            codes.resize(codes_count * code_size);
        }

        // Encode vectors
        #pragma omp parallel num_threads(thread_count)
        {
            unsigned off;
            unsigned cnt;
            compute_thread_task(count, off, cnt);
            long code_off = (labels_offset + off) * code_size;
            pq->encode_multiple_vectors(vectors + off * pq->dim,
                    codes.data() + code_off, cnt);
        }
    }

    template<typename Archive>
    inline void save(Archive& ar) const {
        ar(pq, codes_count, codes);
    }

    template<typename Archive>
    inline void load(Archive& ar) {
        ar(pq, codes_count, codes);
    }
};

void substract_vectors(float *vectors, int dim, int count,
        const float *base_vectors, int *assignements);

void substract_vectors_from_unique(const float* vector, int dim,
        const float* base_vectors, int* assignements, int assign_count,
        float* substracted);

struct index_db: public base_db {
    int part_count;
    std::unique_ptr<float[]> centroids;
    std::unique_ptr<std::vector<std::uint8_t>[]> partitions;
    std::unique_ptr<std::vector<unsigned>[]> labels;

    index_db() {};

    index_db(std::unique_ptr<base_pq>&& pq_, int partition_count_,
            std::unique_ptr<float[]>&& centroids_) :
            base_db(std::move(pq_)), part_count(partition_count_), centroids(
                    std::move(centroids_)) {
        setup_partitions();
    }

    virtual void print(std::ostream& os) const {
        os << "Indexed DB (partitions=" << part_count << ")" << std::endl;
        os << *pq;
    }

    void setup_partitions() {
        partitions.reset(new std::vector<std::uint8_t>[part_count]);
        labels.reset(new std::vector<unsigned>[part_count]);
    }

    virtual void assign_compute_residuals(const float* vector, int multiple_assign,
            int* assignements, float* residuals) {
        // Assign
        const int dim = pq->dim;
        const int count = 1;
        find_k_neighbors(count, part_count, dim,
                multiple_assign, vector, centroids.get(), assignements);
        // Compute residuals
        substract_vectors_from_unique(vector, pq->dim, centroids.get(),
                assignements, multiple_assign, residuals);
    }

    virtual void assign_compute_residuals_mutiple(const float* vectors,
            const int count, const int multiple_assign, int* assignements,
            float* residuals) {
        const int dim = pq->dim;

        // Assign
        find_k_neighbors(count, part_count, dim,
                multiple_assign, vectors, centroids.get(), assignements);

        // Compute residuals
        const int res_dim = multiple_assign * dim;
        for(int vec_i = 0; vec_i < count; ++vec_i) {
            substract_vectors_from_unique(vectors, dim, centroids.get(),
                    assignements, multiple_assign, residuals);
            vectors += dim;
            residuals += res_dim;
            assignements += multiple_assign;
        }
    }

    virtual int partition_count() const {
        return part_count;
    }

    virtual void get_partition(int part_i, const std::uint8_t*& codes_,
            unsigned*& labels_, unsigned& codes_count_) const {
        assert(part_i < part_count);
        codes_ = partitions[part_i].data();
        labels_ = labels[part_i].data();
        codes_count_ = labels[part_i].size();
    }

    virtual void free_partition(int part_i) {
        partitions[part_i].resize(0);
        partitions[part_i].shrink_to_fit();
        labels[part_i].resize(0);
        labels[part_i].shrink_to_fit();
    }

    void assign_single_compute_residuals(float* vectors, unsigned count,
            int* assignements, int thread_count = 1) {

        // Compute residuals
        #pragma omp parallel num_threads(thread_count)
        {
            unsigned off;
            unsigned cnt;
            compute_thread_task(count, off, cnt);
            // Assign
            find_k_neighbors(cnt, part_count, pq->dim, 1, vectors + off * pq->dim,
                             centroids.get(), assignements + off);

            substract_vectors(vectors + off * pq->dim, pq->dim, cnt,
                    centroids.get(), assignements + off);
        }
    }

    virtual void add_vectors(float* vectors, unsigned count,
            unsigned labels_offset, int thread_count = 1) {
        // Assign and compute residuals
        std::unique_ptr<int[]> assignements = std::make_unique<int[]>(count);
        assign_single_compute_residuals(vectors, count, assignements.get(),
                thread_count);

        // Encode
        const long code_size = pq->code_size();
        std::unique_ptr<std::uint8_t[]> codes_buffer = std::make_unique<
                std::uint8_t[]>(count * code_size);

        #pragma omp parallel num_threads(thread_count)
        {
            unsigned off;
            unsigned cnt;
            compute_thread_task(count, off, cnt);
            pq->encode_multiple_vectors(vectors + off * pq->dim,
                    codes_buffer.get() + off * code_size, cnt);
        }
        // Dispatch
        for (unsigned vec_i = 0; vec_i < count; ++vec_i) {
            const int part_i = assignements[vec_i];
            const std::uint8_t* code = codes_buffer.get() + vec_i * code_size;
            partitions[part_i].insert(partitions[part_i].end(), code,
                    code + code_size);
            labels[part_i].push_back(vec_i + labels_offset);
        }
    }

    template<typename Archive>
    inline void save(Archive& ar) const {
        ar(part_count, pq);
        const int centroids_dim = part_count * pq->dim;
        ar(
                cereal::binary_data(centroids.get(),
                        centroids_dim * sizeof(*centroids.get())));
        for (int part_i = 0; part_i < part_count; ++part_i) {
            ar(partitions[part_i]);
        }
        for (int part_i = 0; part_i < part_count; ++part_i) {
            ar(labels[part_i]);
        }
    }

    template<typename Archive>
    inline void load(Archive& ar) {
        ar(part_count, pq);
        const int centroids_dim = part_count * pq->dim;
        centroids.reset(new float[centroids_dim]);
        ar(
                cereal::binary_data(centroids.get(),
                        centroids_dim * sizeof(*centroids.get())));
        setup_partitions();
        for (int part_i = 0; part_i < part_count; ++part_i) {
            ar(partitions[part_i]);
        }
        for (int part_i = 0; part_i < part_count; ++part_i) {
            ar(labels[part_i]);
        }
    }
};

#include <cereal/archives/binary.hpp>
CEREAL_REGISTER_TYPE(flat_db);
CEREAL_REGISTER_TYPE(index_db);
CEREAL_REGISTER_POLYMORPHIC_RELATION(base_db, flat_db);
CEREAL_REGISTER_POLYMORPHIC_RELATION(base_db, index_db);

std::ostream& operator<<(std::ostream& os, const base_db& db);
std::unique_ptr<float[]> learn_coarse_quantizer(
        vectors_owner<float>& learn_vectors, int centroid_count);

void check_db_filename(const std::unique_ptr<flat_db>& db,
        const char* db_filename, const char* db_ext);

#endif /* DATABASES_HPP_ */
