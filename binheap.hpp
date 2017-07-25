//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef BINHEAP_HPP_
#define BINHEAP_HPP_

#include <memory>
#include <algorithm>
#include <functional>

template<typename KeyType, typename ValueType>
class kv_binheap {
private:
    std::unique_ptr<KeyType[]>keys_;
    std::unique_ptr<ValueType[]> values_;
    int capacity_;
    int size_;

    std::unique_ptr<int[]> permutation_array() {
        std::unique_ptr<int[]> perms = std::make_unique<int[]>(size_);
        for(int i = 0; i < size_; ++i) {
            perms[i] = i;
        }
        return perms;
    }

    bool comparator(int index_a, int index_b) {
        return values_[index_a] < values_[index_b];
    }

public:
        kv_binheap():
             keys_(nullptr), values_(nullptr), capacity_(0), size_(0) {}

    kv_binheap(int capacity):
        capacity_(capacity), size_(0) {
        keys_.reset(new KeyType[capacity_]);
        values_.reset(new ValueType[capacity_]);
    }

        void reset_capacity(int capacity) {
            capacity_ = capacity;
            size_ = 0;
            keys_.reset(new KeyType[capacity_]);
            values_.reset(new ValueType[capacity_]);
        }

    int capacity() const {
        return capacity_;
    }

    int size() const {
        return size_;
    }

    ValueType max() const {
        return values_[0];
    }

    const KeyType* keys() {
        return keys_.get();
    }

    const ValueType* values() {
        return values_.get();
    }

    void push(KeyType key, ValueType value) {

        // Binary heap is not full
        if (size_ != capacity_) {
            int index = size_++;
            values_[index] = value;
            keys_[index] = key;
            int parent_index = (index - 1) / 2;
            while (index != 0 && values_[index] > values_[parent_index]) {
                std::swap(values_[index], values_[parent_index]);
                std::swap(keys_[index], keys_[parent_index]);
                index = parent_index;
                parent_index = (index - 1) / 2;
            }
            return;
        }

        // Do we need to do a replace top operation ?
        if (value < values_[0]) {
            int index = 0;
            values_[index] = value;
            keys_[index] = key;
            while (true) {
                const int left_child_index = 2 * index + 1;
                const int right_child_index = 2 * index + 2;
                int largest_child_index;
                if(left_child_index >= size_) {
                    break;
                }
                largest_child_index = left_child_index;
                if(right_child_index < size_ && values_[right_child_index] > values_[left_child_index]) {
                    largest_child_index = right_child_index;
                }
                if(values_[largest_child_index] <= values_[index]) {
                    break;
                }
                std::swap(values_[index], values_[largest_child_index]);
                std::swap(keys_[index], keys_[largest_child_index]);
                index = largest_child_index;
            }
        }
    }

    void sort(KeyType keys[], ValueType values[]) {
        std::unique_ptr<int[]> perms = permutation_array();
        auto comp = std::bind(&kv_binheap::comparator, this,
                std::placeholders::_1, std::placeholders::_2);
        std::sort(perms.get(), perms.get()+size_, comp);
        for(int i = 0; i < size_; ++i) {
            keys[i] = keys_[perms[i]];
            values[i] = values_[perms[i]];
        }
    }

    void sort_keys(KeyType keys[]) {
        std::unique_ptr<int[]> perms = permutation_array();
        auto comp = std::bind(&kv_binheap::comparator, this,
                std::placeholders::_1, std::placeholders::_2);
        std::sort(perms.get(), perms.get()+size_, comp);
        for(int i = 0; i < size_; ++i) {
            keys[i] = keys_[perms[i]];
        }
    }

        void reset() {
            size_ = 0;
        }
};

#endif /* BINHEAP_HPP_ */
