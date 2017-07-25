//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include <iostream>
#include <unistd.h>
#include "databases.hpp"
#include "binheap.hpp"
#include "query_common.hpp"

struct scanner_simple {
    base_db* db;
    scan_func scan;

    void prepare_database(base_db& database) {
        db = &database;
        scan = get_scan_func(*(database.pq));
    }

    typedef kv_binheap<unsigned, float> BhType;
    void query_scan(const float* query,
            int* assign, int ma, float* tables, int table_dim,
            BhType& bh, query_metrics& metrics) {

        // Fill binary heap
        for (int t = 0; t < bh.capacity(); ++t) {
            bh.push(0, std::numeric_limits<float>::max() - t);
        }

        const std::uint8_t* codes;
        unsigned* labels;
        unsigned count;
        for (int ass_i = 0; ass_i < ma; ++ass_i) {
            const int part_i = assign[ass_i];
            db->get_partition(part_i, codes, labels, count);
            scan(codes, labels, count, tables, bh);
            tables += table_dim;
        }
    }
};

void usage() {
    std::cerr << "Usage: db_query: [-r R] [-m MA] [-b BATCH_SIZE] "
            << "[db_file] [query_file] [groundtruth_file]" << std::endl;
    std::exit(1);
}

struct cmdargs: query_args {
    int batch_size;
};

void parse_args(cmdargs& args, int argc, char* argv[]) {
    int opt;
    args.ma = 1;
    args.r = 100;
    args.batch_size = 1;
    while ((opt = getopt(argc, argv, "r:m:b:")) != -1) {
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

    std::unique_ptr<scanner_simple> scanner =
            std::make_unique<scanner_simple>();
    std::unique_ptr<base_centroids_getter> cg = std::make_unique<base_centroids_getter>(
            db.pq.get());

    if (args.batch_size != 1) {
        // Setup engine
        nns_engine_batch<scanner_simple> engine(std::move(scanner),
                std::move(cg), db, args.ma, args.batch_size);

        // Process queries
        process_queries<nns_engine_batch<scanner_simple>, scanner_simple::BhType>(
                args, db, engine, total_metrics, total_recall);
    } else {
        // Setup engine
        nns_engine<scanner_simple> engine(std::move(scanner), std::move(cg), db,
                args.ma);

        // Process queries
        process_queries<nns_engine<scanner_simple>, scanner_simple::BhType>(
                args, db, engine, total_metrics, total_recall);
    }

    // Display
    std::cout << "r,recall,ma,adc_type," << total_metrics.header_string
            << std::endl;
    std::cout << args.r << "," << total_recall << "," << args.ma << ","
            << "adc," << total_metrics << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line args
    cmdargs args;
    parse_args(args, argc, argv);

    // Load database
    std::unique_ptr<base_db> db = load_database(args);
    std::cerr << *db << std::endl;

    // Process queries
    process_queries(args, *db);
}
