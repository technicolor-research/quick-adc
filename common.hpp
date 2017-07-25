//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cstdint>
#include <sys/time.h>

static inline std::uint64_t ustime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (std::uint64_t) tv.tv_sec * 1000000 + tv.tv_usec;
}

#endif /* COMMON_HPP_ */
