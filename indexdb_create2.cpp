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
#include <fstream>
#include <unistd.h>
#include <cereal/archives/binary.hpp>
#include "databases.hpp"

struct cmdargs {
    const char* in_db_filename;
    const char* out_db_filename;
    const char* data_filename;
};

static void usage() {
    std::cerr
            << "Usage: indexdb_create2 [in_db_file] [pqdata_file] [out_db_file]"
            << std::endl;
    std::exit(1);
}

void parse_args(cmdargs& args, int argc, char* argv[]) {

    if (argc < 4) {
        usage();
    }

    args.in_db_filename = argv[1];
    args.data_filename = argv[2];
    args.out_db_filename = argv[3];
}

static std::unique_ptr<base_db> create_final_database(cmdargs& args) {
    // Load database
    std::unique_ptr<base_db> db;
    std::ifstream in_file(args.in_db_filename);
    cereal::BinaryInputArchive in_archive(in_file);
    in_archive(db);

    // Compute Cheap PQ if needed otherwise only load it
    std::unique_ptr<base_pq> pq = pq_from_data_file(args.data_filename);
    db->pq.swap(pq);

    return db;
}

static void save_database(cmdargs& args, std::unique_ptr<base_db>& db) {
    std::ofstream out_file(args.out_db_filename);
    cereal::BinaryOutputArchive out_archive(out_file);
    out_archive(db);
}

int main(int argc, char* argv[]) {

    // Parse command line arguments
    cmdargs args;
    parse_args(args, argc, argv);

    // Create database
    std::unique_ptr<base_db> db = create_final_database(args);

    // Save database
    save_database(args, db);
}
