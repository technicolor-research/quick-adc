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
#include <cmath>
#include <iostream>
#include <memory>
#include <cereal/archives/binary.hpp>
#include "vector_io.hpp"
#include "databases.hpp"

struct cmdargs {
    int centroid_count;
    const char* learn_filename;
    const char* db_filename;
    const char* residuals_filename;
};

static void usage() {
    std::cerr << "Usage: indexdb_create1 [centroid_count] [learn_file] "
            << "[db_file] [residuals_file]" << std::endl;
    std::exit(1);
}

static void parse_args(cmdargs& args, int argc, char* argv[]) {
    if(argc < 5) {
        usage();
    }
    args.centroid_count = std::atoi(argv[1]);
    args.learn_filename = argv[2];
    args.db_filename = argv[3];
    args.residuals_filename = argv[4];
}

void check_assignements(int centroid_count, int* assignements, int count) {
    std::unique_ptr<int[]> hist = std::make_unique<int[]>(centroid_count);
    std::fill(hist.get(), hist.get() + centroid_count, 0);

    for(int vec_i = 0; vec_i < count; ++vec_i) {
        hist[assignements[vec_i]]++;
    }

    for(int cent_i = 0; cent_i < centroid_count; ++cent_i) {
        std::cout << hist[cent_i] << std::endl;
    }
}

void check_residuals(vectors_owner<float>& residuals, const float* backup,
        const int* assignments, const float* centroids) {
    int dim = residuals.dimension;
    for(int vec_i = 0; vec_i < residuals.count; ++vec_i) {
        const float* vec = backup + vec_i * dim;
        const float* residual = residuals.get(vec_i);
        const float* cent = centroids + assignments[vec_i] * dim;
        for(int i = 0; i < dim; ++i) {
            if (std::abs(vec[i] - (cent[i] + residual[i])) > 1e-5) {
                std::cerr << "Residual error: " << vec_i << " " << i << std::endl;
                std::exit(1);
            }
        }
    }
}

/*
 void check_residuals()
 */

static std::unique_ptr<base_db> create_database(
        vectors_owner<float>& learn_vectors, int centroid_count) {

    // Backup
    const unsigned full_dim = learn_vectors.count * learn_vectors.dimension;
    std::unique_ptr<float[]> backup = std::make_unique<float[]>(full_dim);
    std::copy(learn_vectors.get(0), learn_vectors.get(0) + full_dim,
            backup.get());

    // Learn coarse quantizer
    std::unique_ptr<float[]> centroids = learn_coarse_quantizer(learn_vectors,
            centroid_count);
    std::unique_ptr<base_pq> pq(new base_pq(8,8,learn_vectors.dimension));

    index_db* idb = new index_db(std::move(pq), centroid_count, std::move(centroids));
    std::unique_ptr<base_db> db(idb);
    std::cerr << "Done K-Means" << std::endl;

    // Compute residuals
    const int thread_count = optimal_thread_count(learn_vectors.count);
    std::unique_ptr<int[]> assignements = std::make_unique<int[]>(learn_vectors.count);
    idb->assign_single_compute_residuals(learn_vectors.get(0),
            learn_vectors.count, assignements.get(), thread_count);

    // Check
    //check_assignements(centroid_count, assignements.get(), learn_vectors.count);
    check_residuals(learn_vectors, backup.get(), assignements.get(),
            idb->centroids.get());

    return db;
}

static void save_database_residuals(cmdargs& args, vectors_owner<float>& learn_vectors,
        const std::unique_ptr<base_db>& db) {

    // Save database
    std::ofstream out_file(args.db_filename);
    cereal::BinaryOutputArchive out_archive(out_file);
    out_archive(db);

    // Save residuals
    save_vectors(learn_vectors, args.residuals_filename);

}

int main(int argc, char* argv[]) {

    // Parse command line arguments
    cmdargs args;
    parse_args(args, argc, argv);

    // Load learn vectors
    vectors_owner<float> learn_vectors = load_vectors_by_extension(args.learn_filename);

    // Create database
    std::unique_ptr<base_db> db = create_database(learn_vectors, args.centroid_count);

    // Save database and residuals
    save_database_residuals(args, learn_vectors, db);
}
