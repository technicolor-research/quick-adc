//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#include <cmath>
#include <fstream>
#include <iostream>
#include "quantizers.hpp"

/** Factories       */
template<typename T>
static inline void read_from_fstream(T& data, std::fstream& file) {
    file.read(reinterpret_cast<char*>(&data), sizeof(T));
}

template<typename T>
static inline void read_from_fstream(T* data, int count, std::fstream& file) {
    file.read(reinterpret_cast<char*>(data), sizeof(T) * count);
}

void read_pq_from_fstream(base_pq& pq, std::fstream& file) {
    read_from_fstream(pq.dim, file);
    read_from_fstream(pq.sq_count, file);
    read_from_fstream(pq.sq_bits, file);
    pq.setup_centroids();
    read_from_fstream(pq.centroids_flat.get(), pq.all_centroids_dim(), file);
}

void pq_from_data_file(const char* data_filename, base_pq& pq) {
    std::fstream data_file(data_filename, std::fstream::in);
    read_pq_from_fstream(pq, data_file);
}

template<typename OPQType>
void opq_from_data_file(const char* data_filename, OPQType& pq) {
    std::fstream data_file(data_filename, std::fstream::in);
    read_pq_from_fstream(pq, data_file);
    pq.setup_rotation();
    read_from_fstream(pq.rotation.get(), pq.dim * pq.dim, data_file);
}

static inline void invalid_data_filename(const char* data_filename) {
    std::cerr << "Invalid data filename: " << data_filename << std::endl;
    std::cerr << "Filename must end with: .pq.data or .opq.data" << std::endl;
    std::exit(1);
}

enum pq_type {
    type_pq, type_opq
};

pq_type parse_data_filename(const char* data_filename) {
    const std::string filename(data_filename);

    // Parse extention (must be .data)
    std::size_t ext_pos = filename.rfind('.');
    if(ext_pos == std::string::npos) {
        invalid_data_filename(data_filename);
    }

    const std::string before_ext = filename.substr(0, ext_pos);
    const std::string ext = filename.substr(ext_pos, std::string::npos);
    if(ext != ".data") {
        invalid_data_filename(data_filename);
    }

    // Parse pq type
    std::size_t pq_pos = before_ext.rfind('.');
    if(pq_pos == std::string::npos) {
        invalid_data_filename(data_filename);
    }
    const std::string pq_type_ext = before_ext.substr(pq_pos, std::string::npos);
    if(pq_type_ext == ".pq") {
        return type_pq;
    } else if(pq_type_ext == ".opq") {
        return type_opq;
    }

    invalid_data_filename(data_filename);
    return type_pq;
}

std::unique_ptr<base_pq> pq_from_data_file(const char* data_filename) {
    pq_type type = parse_data_filename(data_filename);
    if(type == type_pq) {
        std::unique_ptr<base_pq> pq_ptr = std::make_unique<base_pq>();
        pq_from_data_file(data_filename, *pq_ptr);
        return pq_ptr;
    } else if(type == type_opq) {
        std::unique_ptr<opq> pq_ptr = std::make_unique<opq>();
        opq_from_data_file(data_filename, *pq_ptr);
        return pq_ptr;
    }

    invalid_data_filename(data_filename);
    return nullptr;
}

/** Display         */
std::ostream& operator<<(std::ostream& os, const base_pq& pq) {
    pq.print(os);
    return os;
}
