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
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>

#include "databases.hpp"
#include "vector_io.hpp"

std::ostream &operator<<(std::ostream &os, const base_db &db) {
    db.print(os);
    return os;
}

void substract_vectors(float* vectors, int dim, int count,
        const float* base_vectors, int* assignements) {
    float* vec = vectors;
    for (int vec_i = 0; vec_i < count; ++vec_i) {
        const float *base_vec = base_vectors + assignements[vec_i] * dim;
        // Pls unroll
        for (int comp_i = 0; comp_i < dim; ++comp_i) {
            vec[comp_i] -= base_vec[comp_i];
        }
        vec += dim;
    }
}

void substract_vectors_from_unique(const float* vector, int dim,
        const float* base_vectors, int* assignements, int assign_count,
        float* substracted) {
    for (int ass_i = 0; ass_i < assign_count; ++ass_i) {
        const float *base_vec = base_vectors + assignements[ass_i] * dim;
        // Pls unroll
        for (int comp_i = 0; comp_i < dim; ++comp_i) {
            substracted[comp_i] = vector[comp_i] - base_vec[comp_i];
        }
        substracted += dim;
    }
}

void kmeans_fast_iterations_thread(vectors_owner<float>& learn_vectors, const int centroid_count,
            int* assignements, float* centroids, const int iter_count) {

    const int dim = learn_vectors.dimension;
    std::unique_ptr<int[]> assign_count(new int[centroid_count]);

    for(int iter = 0; iter < iter_count; ++iter) {
        std::cerr << "\rK-Means iteration: " << (iter + 1) << "/" << iter_count;
        std::cerr.flush();

        // Assign to closest centroid
        #pragma omp parallel
        {
            unsigned off;
            unsigned cnt;
            compute_thread_task(learn_vectors.count, off, cnt);
            find_k_neighbors(cnt, centroid_count, dim,
                1, learn_vectors.get(off), centroids, assignements + off);
        }

        // Update centroids
        std::fill_n(centroids, dim * centroid_count, 0);
        std::fill_n(assign_count.get(), centroid_count, 0);
        for(int vec_i = 0; vec_i < learn_vectors.count; ++vec_i) {
            const float* vec = learn_vectors.get(vec_i);
            const int cent_i = assignements[vec_i];
            assign_count[cent_i]++;
            float* centroid = centroids + cent_i * dim;
            for(int comp_i = 0; comp_i < dim; ++comp_i) {
                centroid[comp_i] += vec[comp_i] ;
            }
        }

        for(int cent_i = 0; cent_i < centroid_count; ++cent_i) {
            float* centroid = centroids + cent_i * dim;
            for(int comp_i = 0; comp_i < dim; ++comp_i) {
                centroid[comp_i] /= assign_count[cent_i];
            }
        }
    }
}

const int kmeans_iter_max = 50;

std::unique_ptr<float[]> learn_coarse_quantizer(
        vectors_owner<float>& learn_vectors, int centroid_count) {

    assert(kmeans_iter_max > 2);

    // New
    const int total_dim = centroid_count * learn_vectors.dimension;
    std::unique_ptr<float[]> centroids(new float[total_dim]);
    std::unique_ptr<int[]> assignements(new int[learn_vectors.count]);
    cv::Mat cv_vectors(learn_vectors.count, learn_vectors.dimension, CV_32F, learn_vectors.get(0));
    cv::Mat cv_centroids(centroid_count, learn_vectors.dimension, CV_32F, centroids.get());
    cv::Mat cv_assignements(learn_vectors.count, 1, CV_32S, assignements.get());
    cv::TermCriteria criteria(CV_TERMCRIT_ITER, 2, 0.0);
    const int kmeans_attempts = 1;
    std::cerr << "\n";
    std::cerr << "K-Means OpenCV initialization (2 iterations)";
    cv::kmeans(cv_vectors, centroid_count, cv_assignements, criteria,
               kmeans_attempts, cv::KMEANS_PP_CENTERS, cv_centroids);
    std::cerr << " done\n";
    kmeans_fast_iterations_thread(learn_vectors, centroid_count, assignements.get(),
                      centroids.get(), kmeans_iter_max - 2);
    std::cerr << "\n";

    return centroids;
}

/** Database files  */
static bool ends_with(const std::string &value, const std::string &ending) {
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static inline void invalid_db_filename(const char *db_filename,
        const char *db_ext) {
    std::cerr << "Invalid db filename: " << db_filename << std::endl;
    std::cerr << "Filename must end with: .pq_tag" << db_ext
            << ", where pq_tag is pq, opq, " << std::endl;
    std::cerr << "pq_cheap or opq_cheap." << std::endl;
    std::exit(1);
}

void check_db_filename(const base_db &db, const char *db_filename,
        const char *db_ext) {
    std::string wanted_end(".");
    wanted_end.append(db.pq->get_tag());
    wanted_end.append(db_ext);
    const std::string filename(db_filename);
    if (!ends_with(filename, wanted_end)) {
        invalid_db_filename(wanted_end.c_str(), db_ext);
    }
}
