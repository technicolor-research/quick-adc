//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include <cassert>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <unistd.h>
#include "databases.hpp"
#include "binheap.hpp"
#include "query_common.hpp"
#include "simd_layout.hpp"
#include "simd_scan.hpp"

typedef decltype(&scan_avx_4<16>) simd_scan_func;
simd_scan_func get_simd_scan_func_epi8(base_pq& pq) {
    assert(pq.sq_bits == 4);

    if(pq.sq_count == 16) {
        return scan_avx_4<16>;
    } else if(pq.sq_count == 32) {
        return scan_avx_4<32>;
    }

    std::cerr << "Unsupported (nsq,nsq_bits) configuration." << std::endl;
    std::cerr << "Supported configurations are: (16,4) (32,4)." << std::endl;
    std::exit(1);
}

template<typename T>
struct QuantizerMAX {
    float min;
    float max;
    float delta;
    T M;

    QuantizerMAX(float min_, float max_): min(min_), max(max_) {
        M = std::numeric_limits<T>::max();
        delta = (max - min) / M;
    }

    void quantize(float val, T& qval) {
        if(val >= max) {
            qval = M;
            return;
        }
        qval = static_cast<T>(((val - min)/delta));
    }

    void quantize_tables(const float* tables, __m128i qtables[], int SQ_COUNT) {
        const int NCENT = 16;
        for (int sq_i = 0; sq_i < SQ_COUNT; ++sq_i) {
            const float* sq_table = tables + sq_i * NCENT;
            T qtable[NCENT];
            for(int cent_i = 0; cent_i < NCENT; ++cent_i) {
                quantize(sq_table[cent_i], qtable[cent_i]);
            }
            qtables[sq_i] = _mm_set_epi8(qtable[15], qtable[14], qtable[13],
                    qtable[12], qtable[11], qtable[10], qtable[9], qtable[8],
                    qtable[7], qtable[6], qtable[5], qtable[4], qtable[3],
                    qtable[2], qtable[1], qtable[0]);
        }
    }
};

struct scanner_4 {
    int part_count;
    float keep;
    bool has_labels;

    // Partition starts
    std::unique_ptr<std::uint8_t[]> starts_flat;
    std::unique_ptr<std::uint8_t*[]> starts;
    std::unique_ptr<unsigned[]> starts_sizes;
    std::unique_ptr<unsigned[]> starts_labels_flat;
    std::unique_ptr<unsigned*[]> starts_labels ;

    // Layout partitions
    std::unique_ptr<std::unique_ptr<std::uint8_t[]>[]> parts;
    std::unique_ptr<std::unique_ptr<unsigned[]>[]> labels;
    std::unique_ptr<unsigned[]> parts_sizes;

    simd_scan_func scan;
    scan_func scan_start;

    scanner_4(float keep_) :
            part_count(0), keep(keep_), has_labels(false), scan(nullptr), scan_start(
                    nullptr) {
    }

    void compute_sizes(base_db& db, unsigned& total_starts_size) {
        assert(part_count > 0);

        starts_sizes.reset(new unsigned[part_count]);
        parts_sizes.reset(new unsigned[part_count]);

        // Setup has_labels
        const std::uint8_t* codes_unused;
        unsigned* labels;
        unsigned size;
        db.get_partition(0, codes_unused, labels, size);
        has_labels = (labels != nullptr);
        total_starts_size = 0;
        for (int part_i = 0; part_i < part_count; ++part_i) {
            db.get_partition(part_i, codes_unused, labels, size);
            if (size == 0) {
                starts_sizes[part_i] = 0;
                parts_sizes[part_i] = 0;
                std::cerr << "Warning: Partition " << part_i << " is empty" << std::endl;
            } else {
                if ((labels != nullptr) != has_labels) {
                    std::cerr
                            << "Cannot prepare database. Some partitions have labels and some have not"
                            << std::endl;
                    std::cerr << part_i << " " << size << std::endl;
                    std::exit(1);
                }
                starts_sizes[part_i] = std::max(1u,
                        static_cast<unsigned>(size * keep));
                total_starts_size += starts_sizes[part_i];
                parts_sizes[part_i] = size;
            }
        }
    }

    void allocate_buffers(base_db& db, const int total_starts_size) {
        const int code_size = db.pq->code_size();

        // Partitions starts
        starts_flat.reset(new std::uint8_t[total_starts_size * code_size]);
        starts.reset(new std::uint8_t*[part_count]);
        starts_labels.reset(new unsigned*[part_count]);

        if(has_labels) {
            starts_labels_flat.reset(new unsigned[total_starts_size]);
        } else {
            starts_labels_flat.reset();
        }

        // Partitions
        parts.reset(new std::unique_ptr<std::uint8_t[]>[part_count]);
        labels.reset(new std::unique_ptr<unsigned[]>[part_count]);

    }

    void copy_interleave_partition(base_db& db, int part_i) {

        // Get original partition
        const int code_size = db.pq->code_size();
        const std::uint8_t* orig_part;
        unsigned* orig_labels;
        unsigned size;
        db.get_partition(part_i, orig_part, orig_labels, size);

        if(size == 0) {
            return;
        }

        // Copy start
        std::copy(orig_part, orig_part + starts_sizes[part_i] * code_size,
                starts[part_i]);

        // Allocate buffer for remainder
        const int SIMD_SIZE = 16;
        long interleaved_size = compute_interleaved_size_4(parts_sizes[part_i], code_size, SIMD_SIZE);
        parts[part_i].reset(new std::uint8_t[interleaved_size]);

        // Interleave remainder
        source_partition src { orig_part, code_size, parts_sizes[part_i] };
        interleave_partition_4(parts[part_i].get(), src, SIMD_SIZE);

        // Handle labels
        if(has_labels) {
            std::copy(orig_labels, orig_labels + starts_sizes[part_i], starts_labels[part_i]);
            labels[part_i].reset(new unsigned[parts_sizes[part_i]]);
            std::copy(orig_labels, orig_labels + parts_sizes[part_i],
                    labels[part_i].get());
        } else {
            starts_labels[part_i] = nullptr;
            labels[part_i] = nullptr;
        }

        db.free_partition(part_i);
    }

    void setup_start_shift(base_db& db, int part_i) {
        starts_labels[part_i] = nullptr;
        if (part_i == 0) {
            starts[0] = starts_flat.get();
            if (has_labels) {
                starts_labels[0] = starts_labels_flat.get();
            }
        } else {
            const int code_size = db.pq->code_size();
            const unsigned prev_size = starts_sizes[part_i - 1];
            starts[part_i] = starts[part_i - 1] + prev_size * code_size;
            if (has_labels) {
                starts_labels[part_i] = starts_labels[part_i - 1] + prev_size;
            }
        }
    }

    void prepare_database(base_db& db) {

        part_count = db.partition_count();
        scan = get_simd_scan_func_epi8(*db.pq);
        scan_start = get_scan_func(*db.pq);

        // Sizes
        unsigned total_starts_size;
        compute_sizes(db, total_starts_size);

        // Allocation
        allocate_buffers(db, total_starts_size);

        // Partitions
        for (int part_i = 0; part_i < part_count; ++part_i) {
            setup_start_shift(db, part_i);
            copy_interleave_partition(db, part_i);
        }
    }

    void query_scan_start(int* assign, int ma, float* tables, int table_dim, kv_binheap<unsigned, float>& bh) {

        bh.push(0, std::numeric_limits<float>::max());

        // Scan partition starts
        float* tables_shifted = tables;
        for(int ass_i = 0; ass_i < ma ; ++ass_i) {
            const int part_i = assign[ass_i];
            scan_start(starts[part_i], starts_labels[part_i],
                    starts_sizes[part_i], tables_shifted, bh);
            tables_shifted += table_dim;
        }
    }

    typedef kv_binheap<unsigned, std::int8_t> BhType;
    void query_scan(const float* query,
            int* assign, int ma, float* tables, int table_dim,
            BhType& bh, query_metrics& metrics) {

        // Scan start
        kv_binheap<unsigned, float> tmp_bh(bh.capacity());
        query_scan_start(assign, ma,  tables, table_dim, tmp_bh);

        const int SQ_COUNT = table_dim / 16;

        // Quantize distance tables
        const int all_tables_dim = ma * table_dim;
        std::unique_ptr<__m128i[]> qtables(new __m128i[ma * SQ_COUNT]);
        float qmin = *std::min_element(tables, tables + all_tables_dim);
        float qmax = tmp_bh.max();

        // Normalize qmin value if needed
        if (qmin < 0) {
            qmin = 0;
            for (int i = 0; i < all_tables_dim; ++i) {
                if (tables[i] < 0) {
                    tables[i] = 0;
                }
            }
        }

        if(qmax > 1e30) {
            std::cerr << "Warning: Max quantization bound too high. Try larger keep value."<< std::endl;
            std::exit(1);
        }

        bh.push(0, std::numeric_limits<std::int8_t>::max());
        QuantizerMAX<std::int8_t> q127(qmin, qmax);

        //// Normal quantization
        float* tables_shifted = tables;
        for(int ass_i = 0; ass_i < ma; ++ass_i) {
            q127.quantize_tables(tables_shifted, qtables.get() + ass_i*SQ_COUNT, SQ_COUNT);
            tables_shifted += table_dim;
        }

        // Scan partitions
        if(has_labels) {
            for(int ass_i = 0; ass_i < ma; ++ass_i) {
                const int part_i = assign[ass_i];
                const int part_size = parts_sizes[part_i];
                if(part_size == 0) {
                    continue;
                }
                scan(parts[part_i].get(), labels[part_i].get(), 0,
                        part_size, qtables.get() + ass_i*SQ_COUNT, bh);
            }
        } else {
            for(int ass_i = 0; ass_i < ma; ++ass_i) {
                const int part_i = assign[ass_i];
                const int part_size = parts_sizes[part_i];
                if(part_size == 0) {
                    continue;
                }
                //std::cout << "Calling " << scan << std::endl;
                scan(parts[part_i].get(), nullptr, 0,
                        part_size, qtables.get() + ass_i*SQ_COUNT, bh);
            }
        }
    }
};

struct cmdargs : query_args {
    float keep;
    int batch_size;
};

void usage() {
    std::cerr << "Usage: db_query_4 [-r R] [-m MA] [-k KEEP_PERCENT] [-b BATCH_SIZE] "
            << "[db_file] [query_file] [groundtruth_file]" << std::endl;
    std::exit(1);
}

void parse_args(cmdargs& args, int argc, char* argv[]) {
    const float ONE_PERCENT = 0.01;
    int opt;
    args.ma = 1;
    args.r = 100;
    args.keep = 1 * ONE_PERCENT;
    args.batch_size = 1;
    while ((opt = getopt(argc, argv, "r:m:b:k:")) != -1) {
        switch (opt) {
        case 'r':
            args.r = std::atoi(optarg);
            break;
        case 'm':
            args.ma = std::atoi(optarg);
            break;
        case 'b':
            args.batch_size = std::atoi(optarg);
            break;
        case 'k':
            args.keep = std::atof(optarg) * ONE_PERCENT;
            break;
        default:
            usage();
        }
    }

    if (argc - optind < 3) {
        usage();
    }

    args.db_file = argv[optind];
    args.query_file = argv[optind + 1];
    args.groundtruth_file = argv[optind + 2];
}

void process_queries(cmdargs& args, base_db& db) {
    query_metrics total_metrics;
    double total_recall = 0;


    std::unique_ptr<base_centroids_getter> cg = std::make_unique<
            base_centroids_getter>(db.pq.get());
    std::unique_ptr<scanner_4> scanner = std::make_unique<scanner_4>(
            args.keep);

    if(args.batch_size != 1) {
        // Setup engine
        nns_engine_batch<scanner_4> engine(std::move(scanner), std::move(cg),
                db, args.ma, args.batch_size);

        // Process queries
        process_queries<nns_engine_batch<scanner_4>, scanner_4::BhType>(args,
                db, engine, total_metrics, total_recall);
    } else {
        // Setup engine
        nns_engine<scanner_4> engine(std::move(scanner), std::move(cg), db,
                                     args.ma);

        // Process queries
        process_queries<nns_engine<scanner_4>, scanner_4::BhType>(
                    args, db, engine, total_metrics, total_recall);
    }

    // Display
    std::cout << "r,recall,ma,adc_type,keep," << total_metrics.header_string
            << std::endl;
    std::cout << args.r << "," << total_recall << "," << args.ma << ","
            << "qadc," << args.keep << "," << total_metrics << std::endl;
}

std::unique_ptr<base_db> load_database_check(cmdargs& args) {
    std::unique_ptr<base_db> db = load_database(args);
    const base_pq& pq = *(db->pq);
    if (pq.sq_bits != 4) {
        std::cerr << "Quantizer must have "
                << " sq_bits=4" << std::endl;
        std::exit(1);
    }
    return db;
}

int main(int argc, char* argv[]) {
    // Parse command line args
    cmdargs args;
    parse_args(args, argc, argv);

    // Load database
    std::unique_ptr<base_db> db = load_database_check(args);

    // Process queries
    process_queries(args, *db);
}
