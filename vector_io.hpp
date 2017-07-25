//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef VECTOR_IO_HPP_
#define VECTOR_IO_HPP_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>


template<typename T>
struct vectors {
    T* data;
    int dimension;
    long count;

    T* get(int vector_i) const {
        return data + vector_i * dimension;
    }

};

template<typename T>
struct vectors_owner {
    std::unique_ptr<T[]> data;
    int dimension;
    long count;

    T* get(int vector_i) const {
        return data.get() + vector_i * dimension;
    }

    vectors<T> slice(int start_i, long count) {
        vectors<T> sliced;
        sliced.data = get(start_i);
        sliced.dimension = dimension;
        sliced.count = count;
        return sliced;
    }
};

template<typename T>
void fstream_check_open(const char* filename, T& file,
        std::ios_base::openmode mode = std::ios_base::in) {
    file.open(filename, mode);
    if (!file) {
        std::cerr << "Could not open " << filename << std::endl;
        std::exit(1);
    }
}

int read_vector_dimension(std::ifstream& infile);

template<typename InType>
long count_vectors(std::ifstream& infile, int dimension) {
    infile.seekg(0, std::ifstream::end);
    long vector_count = infile.tellg()
            / (dimension * sizeof(InType) + sizeof(dimension));
    infile.seekg(0, std::ifstream::beg);
    return vector_count;
}

void check_dimension(int read_dimension, int reference_dimension,
        long vector_i);

template<typename InType>
void load_vector_data(std::ifstream& infile, InType* data, int dim,
        long count) {

    for (long vector_i = 0; vector_i < count; ++vector_i) {
        // Read dimension
        int dimension = read_vector_dimension(infile);
        check_dimension(dimension, dim, vector_i);
        // Read data
        infile.read(reinterpret_cast<char*>(data + vector_i * dim),
                sizeof(InType) * dim);
    }
}

template<typename InType>
vectors_owner<InType> load_vectors(const char* filename) {
    vectors_owner<InType> vecs;
    // Open file
    std::ifstream infile;
    fstream_check_open(filename, infile,
            std::ifstream::in | std::ifstream::binary);

    // Allocate buffer
    vecs.dimension = read_vector_dimension(infile);
    vecs.count = count_vectors<InType>(infile, vecs.dimension);
    vecs.data = std::make_unique<InType[]>(vecs.count * vecs.dimension);

    // Read all vectors
    load_vector_data(infile, vecs.data.get(), vecs.dimension, vecs.count);

    return vecs;
}

template<typename InType, typename OutType>
void load_vector_data_convert(std::ifstream& infile, OutType* data, int dim,
        long count) {
    std::unique_ptr<InType[]> temp_vector = std::make_unique<InType[]>(dim);

    for (long vector_i = 0; vector_i < count; ++vector_i) {
        // Read dimension
        int dimension = read_vector_dimension(infile);
        check_dimension(dimension, dim, vector_i);
        // Read data
        infile.read(reinterpret_cast<char*>(temp_vector.get()),
                sizeof(InType) * dim);
        // Copy and (implicitely) cast
        std::copy(temp_vector.get(), temp_vector.get() + dim,
                data + vector_i * dim);
    }
}

template<typename InType, typename OutType>
vectors_owner<OutType> load_vectors_convert(const char* filename) {
    vectors_owner<OutType> vecs;
    // Open file
    std::ifstream infile;
    fstream_check_open(filename, infile,
            std::ifstream::in | std::ifstream::binary);

    // Allocate final buffer
    vecs.dimension = read_vector_dimension(infile);
    vecs.count = count_vectors<InType>(infile, vecs.dimension);
    vecs.data = std::make_unique<OutType[]>(vecs.count * vecs.dimension);

    // Read all vectors
    load_vector_data_convert<InType, OutType>(infile, vecs.data.get(), vecs.dimension,
            vecs.count);
    return vecs;
}

vectors_owner<float> load_vectors_by_extension(const char* filename);

template<typename OutType>
void save_vectors(const vectors_owner<OutType>& vecs, const char* filename) {
    // Open file
    std::ofstream outfile;
    fstream_check_open(filename, outfile, std::ios_base::out);

    // Write vectors
    std::int32_t dimension = vecs.dimension;
    for (int vector_i = 0; vector_i < vecs.count; ++vector_i) {
        outfile.write(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        outfile.write(reinterpret_cast<char*>(vecs.get(vector_i)),
                vecs.dimension * sizeof(OutType));
    }
}

template<typename T>
struct vectors_chunk {
    unsigned count;
    unsigned offset;
    std::unique_ptr<T[]> data;

    vectors_chunk() :
            count(0), offset(0), data(nullptr) {
    }

    vectors_chunk(int dim, unsigned count_, unsigned offset_) :
            count(count_), offset(offset_), data(new T[count * dim]) {
    }

};

typedef decltype(load_vector_data<float>)* load_func_t;

template<typename T>
class safe_bounded_queue {
    std::mutex mutex_;
    std::condition_variable condition_empty_;
    std::condition_variable condition_full_;
    std::queue<T> queue_;
    std::size_t max_size_;

public:
    safe_bounded_queue(int max_size) :
            max_size_(max_size) {
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void push(T&& item) {
        // Wait until queue is not full
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() == max_size_) {
            condition_full_.wait(lock);
        }
        queue_.push(std::move(item));
        lock.unlock();
        condition_empty_.notify_one();
    }

    void pop(T& item) {
        // Wait until queue is not empty
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty()) {
            condition_empty_.wait(lock);
        }
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        condition_full_.notify_one();
    }
};

class vectors_reader {
public:
    // Threads management
    const static int MAX_QUEUE_SIZE = 2;
    safe_bounded_queue<vectors_chunk<float>> queue_;
    // Vectors
    unsigned wanted_chunk_count_;
    int dim_;
    unsigned count_;
    unsigned read_count_;
    // File
    std::string filename_;
    load_func_t load_func_;

    vectors_reader(const char* filename, int dim, unsigned count,
            load_func_t load_func, int chunk_count = 1000000) :
            queue_(MAX_QUEUE_SIZE), wanted_chunk_count_(chunk_count), dim_(dim), count_(
                    count), read_count_(0), filename_(filename), load_func_(
                    load_func) {
        read_count_ = 0;
    }

    void run() {
        std::ifstream infile;
        fstream_check_open(filename_.c_str(), infile,
                std::ifstream::in | std::ifstream::binary);
        while (read_count_ != count_) {
            // Read chunk
            unsigned chunk_count = std::min(wanted_chunk_count_,
                    count_ - read_count_);
            vectors_chunk<float> chunk(dim_, chunk_count, read_count_);
            load_func_(infile, chunk.data.get(), dim_, chunk.count);

            // Push chunk
            read_count_ += chunk_count;
            queue_.push(std::move(chunk));
        }
        infile.close();
        std::cerr << "Vector reader exited" << std::endl;
    }

    unsigned count() const {
        return count_;
    }

    unsigned read_count() const {
        return read_count_;
    }

    int dim() const {
        return dim_;
    }

    bool done() {
        return (count_ == read_count_) && queue_.empty();
    }

    vectors_chunk<float> get_chunk() {
        vectors_chunk<float> chunk;
        queue_.pop(chunk);
        return chunk;
    }
};

std::unique_ptr<vectors_reader> vectors_reader_by_extension(const char* filename);

#endif /* VECTOR_IO_HPP_ */
