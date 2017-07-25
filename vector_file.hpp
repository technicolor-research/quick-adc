//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef VECTOR_FILE_HPP_
#define VECTOR_FILE_HPP_

#include <cstdint>
#include <cstdarg>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sendfile.h>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>

int open_or_die(const char* filename, int flags) {
    int fd = open(filename, flags);
    if(fd == -1) {
        std::stringstream ss;
        ss << "Could not open file " << filename;
        perror(ss.str().c_str());
        exit(1);
    }
    return fd;
}

int open_or_die(const char* filename, int flags, mode_t mode) {
    int fd = open(filename, flags, mode);
    if(fd == -1) {
        std::stringstream ss;
        ss << "Could not open file " << filename;
        perror(ss.str().c_str());
        exit(1);
    }
    return fd;
}

void read_or_die(int fd, void* buf, size_t sz) {
    ssize_t ssz = static_cast<size_t>(sz);
    if(read(fd, buf, sz) != ssz) {
        std::stringstream ss;
        ss << "Could not read " << sz << " bytes from file";
        perror(ss.str().c_str());
        exit(1);
    }
}

void write_or_die(int fd, const void* buf, size_t sz) {
    ssize_t ssz = static_cast<size_t>(sz);
    if(write(fd, buf, sz) != ssz) {
        std::stringstream ss;
        ss << "Could not write " << sz << " bytes from file";
        perror(ss.str().c_str());
        exit(1);
    }
}

void filename_error(const char* filename) {
    std::cerr << "Could not load vectors from " << filename << std::endl;
    std::cerr << "Unknown extension\n";
    std::cerr << "Known extensions: .bvecs, .ivecs, .fvecs" << std::endl;
    exit(1);
}

int extension_to_component_size(const char* extension, const char* filename) {

    if (!strcmp(extension, ".bvecs")) {
        return 1;
    } else if (!strcmp(extension, ".fvecs") || !strcmp(extension, ".ivecs")) {
        return 4;
    }

    filename_error (filename);
    return 0;
}

template<typename ... Args>
const char* check_extensions(const char* in_filename, Args ... filenames) {
    const char* in_extension = strrchr(in_filename, '.');

    const char* out_filenames[] = { filenames... };

    for (auto&& out_filename : out_filenames) {
        if (auto out_extension = strrchr(out_filename, '.')) {
            if (strcmp(in_extension, out_extension)) {
                std::cerr
                        << "Input file and output files must have the same extension."
                        << std::endl;
                exit(1);
            }
        } else {
            filename_error(out_filename);
        }
    }

    return in_extension;
}

void sendfile_loop(int out_fd, int in_fd, off_t& offset, const size_t size) {
    const off_t end = offset + size;
    off_t remaining = size;
    while(remaining > 0) {
        const ssize_t ret = sendfile(out_fd, in_fd, &offset, remaining);
        if(ret < 0) {
            std::cerr << "Error while copying data to destination file." << std::endl;
            exit(1);
        }
        remaining = end - offset;
    }
}

std::int32_t read_dimension_rewind(int in_fd) {
    std::int32_t dim;
    if(read(in_fd, reinterpret_cast<void*>(&dim), sizeof(dim)) != sizeof(dim)) {
        std::cerr << "Could not read vector dimension" << std::endl;
        exit(1);
    }
    lseek(in_fd, 0, SEEK_SET);
    return dim;
}

size_t tell_size_rewind(int in_fd) {
    off_t end = lseek(in_fd, 0, SEEK_END);
    lseek(in_fd, 0, SEEK_SET);
    return static_cast<size_t>(end);
}

#endif /* VECTOR_FILE_HPP_ */
