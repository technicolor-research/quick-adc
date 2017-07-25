//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include "vector_io.hpp"

int read_vector_dimension(std::ifstream& infile) {
    std::int32_t read_dimension;
    infile.read(reinterpret_cast<char*>(&read_dimension),
            sizeof(read_dimension));
    return static_cast<int>(read_dimension);
}

void check_dimension(int read_dimension, int reference_dimension,
        long vector_i) {
    if (read_dimension != reference_dimension) {
        std::cerr << "Error while reading vectors." << std::endl;
        std::cerr << "Vector " << vector_i << " has " << read_dimension
                << " dimensions while other vectors have "
                << reference_dimension << " dimensions" << std::endl;
        std::cerr << "All vectors must have the same number of " << "dimensions"
                << std::endl;
        std::exit(1);
    }
}

static void filename_error(const char* filename) {
    std::cerr << "Could not load vectors from " << filename << std::endl;
    std::cerr << "Unknown extension\n";
    std::cerr << "Known extensions: .bvecs, .ivecs, .fvecs" << std::endl;
    std::exit(1);
}

vectors_owner<float> load_vectors_by_extension(const char* filename) {
    vectors_owner<float> vecs;
    const char* extension = strrchr(filename, '.');

    if(extension == nullptr) {
        filename_error(filename);
    }

    if (!strcmp(extension, ".bvecs")) {
        return load_vectors_convert<std::uint8_t, float>(filename);
    } else if (!strcmp(extension, ".fvecs")) {
        return load_vectors<float>(filename);
    } else if(!strcmp(extension, ".ivecs")) {
        return load_vectors_convert<int, float>(filename);
    }

    filename_error(filename);
    return vecs;
}

std::unique_ptr<vectors_reader> vectors_reader_by_extension(
        const char* filename) {
    const char* extension = strrchr(filename, '.');

    if (extension == nullptr) {
        filename_error(filename);
    }

    std::ifstream infile;
    fstream_check_open(filename, infile,
            std::ifstream::in | std::ifstream::binary);
    int dim = read_vector_dimension(infile);
    if (!strcmp(extension, ".bvecs")) {
        load_func_t load_func = load_vector_data_convert<std::uint8_t, float>;
        long count = count_vectors<std::uint8_t>(infile, dim);
        return std::make_unique<vectors_reader>(filename, dim, count,
                load_func);
    } else if (!strcmp(extension, ".fvecs")) {
        load_func_t load_func = load_vector_data<float>;
        long count = count_vectors<float>(infile, dim);
        return std::make_unique<vectors_reader>(filename, dim, count,
                load_func);
    } else if (!strcmp(extension, ".ivecs")) {
        load_func_t load_func = load_vector_data_convert<int, float>;
        long count = count_vectors<int>(infile, dim);
        return std::make_unique<vectors_reader>(filename, dim, count,
                load_func);
    }

    filename_error(filename);
    return std::make_unique<vectors_reader>(filename, 0, 0, nullptr);
}
