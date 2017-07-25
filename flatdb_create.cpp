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
#include <fstream>
#include <unistd.h>
#include <cereal/archives/binary.hpp>
#include "quantizers.hpp"
#include "databases.hpp"

struct cmdargs {
    const char* data_filename;
    const char* db_filename;
};

void usage() {
    std::cerr << "Usage: flatdb_create [pqdata_file] [db_file]"
            << std::endl;
    std::exit(1);
}

void parse_args(cmdargs& args, int argc, char* argv[]) {

    if (argc < 3) {
        usage();
    }

    args.data_filename = argv[1];
    args.db_filename = argv[2];
}

static std::unique_ptr<base_db> create_flat_database(cmdargs& args) {

    // Load PQ file
    std::unique_ptr<base_pq> pq = pq_from_data_file(args.data_filename);

    // Create database
    std::unique_ptr<flat_db> db = std::make_unique<flat_db>(std::move(pq));
    return db;
}

static void save_database(cmdargs& args, const std::unique_ptr<base_db>& db) {
    std::ofstream out_file(args.db_filename);
    cereal::BinaryOutputArchive out_archive(out_file);
    out_archive(db);
}

int main(int argc, char* argv[]) {

    // Parse command line arguments
    cmdargs args;
    parse_args(args, argc, argv);

    // Create database
    std::unique_ptr<base_db> db = create_flat_database(args);

    // Save database
    save_database(args, db);

}
