//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef NEIGHBORS_HPP_
#define NEIGHBORS_HPP_

void find_k_neighbors(
        const int vector_count, const int neighbor_count, const int dim, 
        const int k, const float* vectors, const float* neighbors, 
        int* assignements);

#endif /* NEIGHBORS_HPP_ */
