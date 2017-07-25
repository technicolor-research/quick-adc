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
#include <iomanip>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <omp.h>
#include <cereal/archives/binary.hpp>
#include "databases.hpp"
#include "vector_io.hpp"
#include "common.hpp"

void usage() {
    std::cerr << "Usage: db_add [db_file] [base_file]" << std::endl;
    std::exit(1);
}

struct cmdargs {
    const char* db_filename;
    const char* base_filename;
};

void parse_args(cmdargs& args, int argc, char* argv[]) {
    if (argc < 3) {
        usage();
    }
    args.db_filename = argv[1];
    args.base_filename = argv[2];
}

std::unique_ptr<base_db> load_database(const cmdargs& args) {
    std::unique_ptr<base_db> db;
    std::ifstream in_file(args.db_filename);
    cereal::BinaryInputArchive in_archive(in_file);
    in_archive(db);
    std::cout << "Loaded database"  << std::endl;
    std::cout << *db << std::endl;
    return db;
}

void add_vectors(const cmdargs& args, base_db& db) {

    std::unique_ptr<vectors_reader> reader = vectors_reader_by_extension(
            args.base_filename);

    vectors_reader* reader_ptr = reader.get();

    std::thread read_thread([reader_ptr] {
        reader_ptr->run();
        std::cerr << "Reader thread exited" << std::endl;
    });

    while (!reader->done()) {
        std::cerr << "Waiting chunk";
        std::cerr.flush();
        vectors_chunk<float> chunk = reader->get_chunk();
        const int thread_count = optimal_thread_count(chunk.count);
        const std::uint64_t start_add_us = ustime();
        std::cerr << "\rChunk: " << chunk.offset << " " << chunk.count
                << " (" << thread_count << " threads) ";
        db.add_vectors(chunk.data.get(), chunk.count, chunk.offset,
                thread_count);
        const float add_s = (ustime() - start_add_us) / 1e6;
        std::ios::fmtflags display_flags(std::cerr.flags());
        std::cerr << "(" << std::fixed << std::setprecision(3) << add_s << "s)"<< std::endl;
        std::cerr.flags(display_flags);
    }

    std::cerr << "Done reading chunks" << std::endl;
    read_thread.join();
}

void save_database(const cmdargs& args, const std::unique_ptr<base_db>& db) {
    std::cerr << "Saving database" << std::endl;
    std::ofstream out_file(args.db_filename,
            std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    cereal::BinaryOutputArchive out_archive(out_file);
    out_archive(db);
}

int main(int argc, char* argv[]) {
    cmdargs args;
    parse_args(args, argc, argv);
    std::unique_ptr<base_db> db = load_database(args);
    add_vectors(args, *db);
    save_database(args, db);
}
