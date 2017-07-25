//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef QUERY_COMMON_HPP_
#define QUERY_COMMON_HPP_

#include <cstdint>
#include "databases.hpp"
#include "distances.hpp"
#include "vector_io.hpp"
#include "recall.hpp"
#include "common.hpp"

struct query_metrics {
    std::uint64_t index_us;
    std::uint64_t rotate_us;
    std::uint64_t table_us;
    std::uint64_t scan_us;
    static constexpr const char* header_string = "index_us,rotate_us,table_us,scan_us";

    query_metrics() :
            index_us(0), rotate_us(0), table_us(0), scan_us(0) {
    }

    query_metrics& operator+=(const query_metrics& rhs) {
        index_us += rhs.index_us;
        rotate_us += rhs.rotate_us;
        table_us += rhs.table_us;
        scan_us += rhs.scan_us;
        return *this;
    }

    query_metrics& operator/=(const int f) {
        index_us /= f;
        rotate_us /= f;
        table_us /= f;
        scan_us /= f;
        return *this;
    }

    virtual ~query_metrics() {};

};

std::ostream& operator<<(std::ostream& os, const query_metrics& metrics) {
    os << metrics.index_us << "," << metrics.rotate_us << ","
            << metrics.table_us << "," << metrics.scan_us;
    return os;
}

/** PQ Scan functions */
template<int NSQ>
void scan_4(const std::uint8_t* pqcodes, const unsigned* labels,
        const unsigned pqcodes_count, const float* dists,
        kv_binheap<unsigned, float>& bh) {

    const int CODE_BYTES = NSQ / 2;
    static_assert(NSQ % 2 == 0, "Invalid NSQ");
    const int NCENT = 1 << 4;

    float min = bh.max();
    float candidate;
    for(unsigned pqcode_i = 0; pqcode_i < pqcodes_count; ++pqcode_i) {
        const uint8_t* const pqcode = pqcodes + pqcode_i * CODE_BYTES;
        candidate = 0;
        // You better unroll this shit, gcc. You better do it.
        for(int byte_i = 0; byte_i < CODE_BYTES; ++byte_i) {
            const int sq_i = 2*byte_i;
            const std::uint8_t comp0 = pqcode[byte_i] & 0xf;
            candidate += dists[sq_i * NCENT + comp0];
            const std::uint8_t comp1 = (pqcode[byte_i] & 0xf0) >> 4;
            candidate += dists[(sq_i + 1) * NCENT + comp1];
        }
        if (candidate < min) {
            if(labels != nullptr) {
                bh.push(labels[pqcode_i], candidate);
            } else {
                bh.push(pqcode_i, candidate);
            }
            min = bh.max();
        }
    }
}

template<typename T, int NSQ>
void scan_standard(const std::uint8_t* pqcodes_, const unsigned* labels,
        const unsigned pqcodes_count, const float* dists,
        kv_binheap<unsigned, float>& bh) {

    const int NCENT = 1 << (sizeof(T) * 8);

    // Scan pqcodes
    const T* const pqcodes = reinterpret_cast<const T*>(pqcodes_);
    float min = bh.max();
    float candidate;
    for (unsigned pqcode_i = 0; pqcode_i < pqcodes_count; ++pqcode_i) {
        const T* const pqcode = pqcodes + pqcode_i * NSQ;
        candidate = 0;
        for (int sq_i = 0; sq_i < NSQ; ++sq_i) {
            candidate += dists[sq_i * NCENT + pqcode[sq_i]];
        }
        if (candidate < min) {
            if(labels != nullptr) {
                bh.push(labels[pqcode_i], candidate);
            } else {
                bh.push(pqcode_i, candidate);
            }
            min = bh.max();
        }
    }
}

typedef decltype(&scan_standard<std::uint8_t, 8>) scan_func;
scan_func get_scan_func(base_pq& pq) {
    if(pq.sq_count == 16 && pq.sq_bits == 4) {
        return scan_4<16>;
    } else if(pq.sq_count == 32 && pq.sq_bits == 4) {
        return scan_4<32>;
    } else if(pq.sq_count == 4 && pq.sq_bits == 8) {
        return scan_standard<std::uint8_t, 4>;
    } else if(pq.sq_count == 8 && pq.sq_bits == 8) {
        return scan_standard<std::uint8_t, 8>;
    } else if(pq.sq_count == 16 && pq.sq_bits == 8) {
        return scan_standard<std::uint8_t, 16>;
    } else if(pq.sq_count == 2 && pq.sq_bits == 16) {
        return scan_standard<std::uint16_t, 2>;
    } else if(pq.sq_count == 4 && pq.sq_bits == 16) {
        return scan_standard<std::uint16_t, 4>;
    } else if(pq.sq_count == 8 && pq.sq_bits == 16) {
        return scan_standard<std::uint16_t, 8>;
    }

    std::cerr << "Unsupported (nsq,nsq_bits) configuration." << std::endl;
    std::cerr << "Supported configurations are: (16,4) (4,8) (8,8) (16,8) (2,16) (4,16) (8,16)." << std::endl;
    std::exit(1);
}

//scan_ = get_scan_function(*db_.pq);

const long TABLES_BUFFER_SIZE = 1 * 1024L * 1024L * 1024L;

template<typename ScannerType>
struct nns_engine_batch {

    base_db& db_;
    std::unique_ptr<centroids_getter> cg_;
    int ma_;
    int batch_count_;
    std::unique_ptr<ScannerType> scanner_;
    std::unique_ptr<int[]> assignements_;
    std::unique_ptr<float[]> residuals_;
    std::unique_ptr<float[]> dists_;
    dists_mutiple_func dist_mult_func_;
    long batch_shift_;


    nns_engine_batch(std::unique_ptr<ScannerType>&& scanner,
            std::unique_ptr<centroids_getter>&& cg,
            base_db& db,
            int ma, int batch_count = -1) :
            db_(db), cg_(std::move(cg)), ma_(ma),
            batch_count_(batch_count), scanner_(std::move(scanner)) {
        assert(ma > 0);
        if (batch_count_ <= 0) {
            const long table_size = cg_->sq_count() * cg_->sq_centroid_count()
                    * sizeof(float);
            batch_count_ = TABLES_BUFFER_SIZE / (ma * table_size);
        }
        std::cerr << "NNS Engine Batch size: " << batch_count_ << " queries" << std::endl;
        assignements_.reset(new int[batch_count_ * ma_]);

        const int table_dim =  cg_->sq_count() * cg_->sq_centroid_count();
        dists_.reset(new float[batch_count_ * ma_ * static_cast<long>(table_dim)]);

        assert(ma_ * batch_count_ > 1);
        dist_mult_func_= get_dists_mutiple_function(db_.pq->sq_dim());

        batch_shift_ = batch_count_ * db_.pq->dim;
        residuals_.reset(new float[batch_shift_ * ma_]);

    }

    void prepare_database() {
        scanner_->prepare_database(db_);
    }

    void batch_process_queries(const int query_i, const float* queries, const int count,
            query_metrics& metrics) {

        const int batch_i = query_i / batch_count_;
        const float* batch_queries = queries + batch_i * batch_shift_;
        const int this_batch_count = std::min(batch_count_,
                count - batch_i * batch_count_);
        const std::uint64_t assign_start_us = ustime();

        db_.assign_compute_residuals_mutiple(batch_queries, this_batch_count, ma_, assignements_.get(),
                residuals_.get());

        const std::uint64_t rotate_start_us = ustime();
        db_.pq->rotate_multiple_vectors(residuals_.get(), this_batch_count * ma_);

        const std::uint64_t tables_start_us = ustime();

        dists_mutiple_func dist_func = get_dists_mutiple_function(db_.pq->sq_dim());
        dist_func(dists_.get(), *cg_, residuals_.get(),
                this_batch_count * ma_);

        metrics.table_us = ustime() - tables_start_us;
        metrics.rotate_us = tables_start_us - rotate_start_us;
        metrics.index_us = rotate_start_us - assign_start_us;
    }

    template<typename DistType, typename MetricsType>
    void process_query(const int query_i, const float* queries, const int count,
            kv_binheap<unsigned, DistType>& bh,
            MetricsType& metrics) {

        const int query_batch_i = query_i % batch_count_;
        if(query_batch_i == 0) {
            batch_process_queries(query_i, queries, count, metrics);
        } else {
            metrics.table_us = 0;
            metrics.rotate_us = 0;
            metrics.index_us = 0;
        }

        const std::uint64_t scan_start_us = ustime();
        const int table_dim = cg_->sq_count() * cg_->sq_centroid_count();
        int* assign = assignements_.get() + query_batch_i * ma_;
        float* tables = dists_.get() + query_batch_i * ma_ * static_cast<long>(table_dim);
        const float* res = residuals_.get() + query_batch_i * ma_ * db_.pq->dim;
        scanner_->query_scan(res, assign, ma_, tables, table_dim, bh, metrics);

        metrics.scan_us = ustime() - scan_start_us;
    }
};

template<typename ScannerType>
struct nns_engine {

    base_db& db_;
    std::unique_ptr<centroids_getter> cg_;
    const int ma_;
    int table_dim_;
    std::unique_ptr<float[]> residuals_;
    std::unique_ptr<int[]> assign_;
    std::unique_ptr<float[]> dists_;
    std::unique_ptr<ScannerType> scanner_;
    dists_func dist_func_;
    dists_mutiple_func dist_mult_func_;

    nns_engine(std::unique_ptr<ScannerType>&& scanner,
            std::unique_ptr<centroids_getter>&& cg, base_db& db, int ma) :
            db_(db), cg_(std::move(cg)), ma_(ma), residuals_(
                    new float[ma * db.pq->dim]), assign_(new int[ma]), scanner_(
                    std::move(scanner)) {
        assert(ma > 0);
        std::cerr << "No batch" << std::endl;
        table_dim_ = cg_->sq_centroid_count() * cg_->sq_count();
        dists_.reset(new float[ma_ * table_dim_]);
        dist_func_ = get_dists_function(db_.pq->sq_dim());
        dist_mult_func_ = get_dists_mutiple_function(db_.pq->sq_dim());
        residuals_.reset(new float[ma_ * db_.pq->dim]);
        assign_.reset(new int[ma_]);
    }

    void prepare_database() {
        scanner_->prepare_database(db_);
    }

    template<typename DistType, typename MetricsType>
    void process_query(const int query_i, const float* queries, const int count,
            kv_binheap<unsigned, DistType>& bh,
            MetricsType& metrics) {

        const float* query = queries + query_i * db_.pq->dim;
        const std::uint64_t index_start_us = ustime();
        db_.assign_compute_residuals(query, ma_, assign_.get(),
                residuals_.get());

        const std::uint64_t rotate_start_us = ustime();
        db_.pq->rotate_multiple_vectors(residuals_.get(), ma_);

        const std::uint64_t tables_start_us = ustime();
        if (ma_ == 1) {
            // Optimized
            dist_func_(dists_.get(), *cg_, residuals_.get());
        } else {
            dist_mult_func_(dists_.get(), *cg_, residuals_.get(), ma_);
        }

        const std::uint64_t scan_start_us = ustime();
        scanner_->query_scan(residuals_.get(), assign_.get(), ma_, dists_.get(),
                table_dim_, bh, metrics);

        metrics.scan_us = ustime() - scan_start_us;
        metrics.table_us = scan_start_us - tables_start_us;
        metrics.rotate_us = tables_start_us - rotate_start_us;
        metrics.index_us = rotate_start_us - index_start_us;
    }

};

/** Utility functions       */

struct query_args {
    const char* db_file;
    const char* query_file;
    const char* groundtruth_file;
    int r;
    int ma;
};

std::unique_ptr<base_db> load_database(query_args& args) {
    std::cerr << "Database file: " << args.db_file << std::endl;
    std::unique_ptr<base_db> db;
    std::ifstream in_file(args.db_file);
    cereal::BinaryInputArchive in_archive(in_file);
    in_archive(db);
    return db;
}

template<typename EngineType, typename BhType, typename MetricsType>
void process_queries(query_args& args, base_db& db, EngineType& engine,
        MetricsType& total_metrics, double& total_recall, int max_queries = -1) {

    // Load queries
    vectors_owner<float> queries = load_vectors_by_extension(args.query_file);
    if(max_queries > 0) {
        queries.count = max_queries;
    }

    // Load groundtruth file
    recall_file rec_file(args.groundtruth_file);
    const int t = 1;

    // Metrics
    MetricsType metrics;

    // Prepare database
    engine.prepare_database();
    const float* queries_buffer = queries.get(0);

    for(int query_i = 0; query_i < queries.count; ++query_i) {
        std::cerr << query_i + 1 << "/" << queries.count << "\r";
        std::cerr.flush();
        BhType bh(args.r);
        engine.process_query(query_i, queries_buffer, queries.count, bh,
                metrics);
        if(bh.size() != args.r) {
            std::cerr << " WARNING: Binheap not full" << std::endl;
        }
        const int check =rec_file.check_labels(query_i, bh.keys(),
                bh.keys() + args.r, t);

        total_recall += check;
        total_metrics += metrics;
    }
    total_metrics /= queries.count;
    total_recall /= queries.count;
}

#endif /* QUERY_COMMON_HPP_ */
