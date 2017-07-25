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
#include <cstring>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include "vector_file.hpp"

struct cmdargs {
    const char* in_filename;
    const char* out_filename;
    long chunk_id;
    long chunk_size;
};

void split_vector_file(const cmdargs& args) {

    // Check file extensions
    const char* ext = check_extensions(args.in_filename, args.out_filename);

    // Open input file
    int in_fd = open_or_die(args.in_filename, O_RDONLY);

    // Read vector dimension
    std::int32_t dim = read_dimension_rewind(in_fd);

    // Compute and display information
    const int comp_size = extension_to_component_size(ext, args.in_filename);
    const long first_vec_id = args.chunk_id * args.chunk_size;
    const unsigned vec_size = dim * comp_size + sizeof(std::int32_t);
    off_t offset = first_vec_id * vec_size;
    const size_t chunk_bytes = args.chunk_size * vec_size ;

    int out_fd = open_or_die(args.out_filename, O_WRONLY | O_CREAT,
            S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

    std::cout << "Vector dimensions: " << dim << ", Vector bytes: " << vec_size << std::endl;
    std::cout << "First vector of chunk: " << first_vec_id << std::endl;
    std::cout << "Chunk vector count: " << args.chunk_size << ", Chunk bytes: " <<
            chunk_bytes << std::endl;

    sendfile_loop(out_fd, in_fd, offset, chunk_bytes);
    close(in_fd);
    close(out_fd);

}

void usage() {
    std::cerr << "Usage: split_vecs [chunk_id] [chunk_size] [in_file] [out_file]" << std::endl;
    exit(1);
}

void parse_args(cmdargs& args, int argc, char* argv[]) {
    if (argc < 5) {
        usage();
    }

    args.chunk_id = atol(argv[1]);
    args.chunk_size = atol(argv[2]);
    args.in_filename = argv[3];
    args.out_filename = argv[4];
}

int main(int argc, char* argv[]) {
    // Parse command line args
    cmdargs args;
    parse_args(args, argc, argv);

    split_vector_file(args);
}
