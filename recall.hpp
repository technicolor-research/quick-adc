//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef RECALL_HPP_
#define RECALL_HPP_

#include <cstdlib>
#include <vector>
#include <algorithm>
#include "binheap.hpp"
#include "vector_io.hpp"

template<typename ForwardIterator1, typename ForwardIterator2>
bool all_in(ForwardIterator1 oracle_first, ForwardIterator1 oracle_last,
        ForwardIterator2 first, ForwardIterator2 last) {
    while (oracle_first != oracle_last) {
        if (std::find(first, last, *oracle_first) == last) {
            return false;
        }
        oracle_first++;
    }
    return true;
}
;

class recall_file {
    vectors_owner<int> groundtruth;

public:
    recall_file(const char* filename) {
        groundtruth = load_vectors<int>(filename);
    }

    int max_t() const {
        return groundtruth.dimension;
    }

    template<typename ForwardIterator>
    int check_labels(const int query_i, ForwardIterator labels_first,
            ForwardIterator labels_last, const int t) const {
        int* groundtruth_query = groundtruth.get(query_i);
        if (all_in(groundtruth_query, groundtruth_query + t, labels_first,
                labels_last)) {
            return 1;
        }
        return 0;
    }

    int check_labels(const int query_i, const std::vector<int>& labels,
            const int t) const {
        return check_labels(query_i, labels.begin(), labels.end(), t);
    }

};

#endif /* RECALL_HPP_ */
