//
// Copyright (c) 2017 – Technicolor R&D France
//
// The source code form of this open source project is subject to the terms of the
// Clear BSD license.
//
// You can redistribute it and/or modify it under the terms of the Clear BSD
// License (See LICENSE file).
//

#ifndef QUANTIZERS_HPP_
#define QUANTIZERS_HPP_

#include <cassert>
#include <iostream>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include "neighbors.hpp"

extern "C" {
#include <cblas.h>
}

#define NEW
static inline float* subv(float* vec, int dim, int i) {
	return vec + (i * dim);
}

static inline void set_bits_generic(std::uint8_t* code, std::uint32_t src, int offset) {
	std::uint32_t* shifted =
			reinterpret_cast<std::uint32_t*>(code + (offset / 8));
	*shifted |= (src) << (offset % 8);
}

template<typename T>
inline void multiple_set_bits_native(int* assign, int count, int sq_count, int sq_bits,
		int sq_i, std::uint8_t* codes) {
	assert(sq_bits % 8 == 0);
	const int sq_bytes = sq_bits / 8;
	assert(sq_bytes == sizeof(T));
	T* native_codes = reinterpret_cast<T*>(codes);
	native_codes += sq_i;
	for(int vec_i = 0; vec_i < count; ++vec_i) {
		*native_codes = static_cast<T>(assign[vec_i]);
		native_codes += sq_count;
	}
}

inline void multiple_set_bits_4(int* assign, int count, int sq_count,
		int sq_bits, int sq_i, std::uint8_t* codes) {
	assert(sq_count % 2 == 0);
	assert(sq_bits == 4);
	const int code_size = sq_count / 2;
	codes += sq_i / 2;
	if (sq_i % 2 == 1) {
		for (int vec_i = 0; vec_i < count; ++vec_i) {
			const std::uint8_t a = static_cast<std::uint8_t>(assign[vec_i]);
			*codes = *codes | (a << 4);
			codes += code_size;
		}
	} else {
		for (int vec_i = 0; vec_i < count; ++vec_i) {
			const std::uint8_t a = static_cast<std::uint8_t>(assign[vec_i]);
			*codes = a;
			codes += code_size;
		}
	}
}

inline decltype(&multiple_set_bits_4) prepare_multiple_set_bits(int sq_bits, std::uint8_t* codes, int size) {
	switch (sq_bits) {
	case 16:
		return multiple_set_bits_native<std::uint16_t> ;
	case 8:
		return multiple_set_bits_native<std::uint8_t> ;
	case 4:
		return multiple_set_bits_4;
	default:
		std::cerr << "Could not get multiple set bits function for sq_bits="
				<< sq_bits << std::endl;
		std::exit(1);
	}
	return nullptr;
}

inline void extract_subvectors(const float* vectors, int dim, int count, int sq_dim,
		int sq_i, float* subvectors) {
	vectors += sq_dim * sq_i;
	for (int vec_i = 0; vec_i < count; ++vec_i) {
		std::copy(vectors, vectors + sq_dim, subvectors);
		subvectors += sq_dim;
		vectors += dim;
	}
}

class base_pq {

public:
	int sq_count;
	int sq_bits;
	int dim;
	std::unique_ptr<float*[]> centroids;
	std::unique_ptr<float[]> centroids_flat;

	base_pq() {};

	base_pq(int sq_count_, int sq_bits_, int dim_, float* centroids_flat_ = nullptr):
			sq_count(sq_count_), sq_bits(sq_bits_), dim(dim_) {

		assert(sq_bits < 30);
		assert(dim % sq_count == 0);
		assert(sq_bits == 4 || sq_bits == 8 || sq_bits == 16);
		assert((sq_count * sq_bits) % 8 == 0);

		setup_centroids();
		const int comp_count = all_centroids_dim();
		if (centroids_flat_ != nullptr) {
			std::copy(centroids_flat_, centroids_flat_ + comp_count,
					centroids_flat.get());
		}
	}

	virtual ~base_pq() {};

	virtual void print(std::ostream& os) const {
		os << get_tag() << " (dim=" << dim << ", sq=" << sq_count << "x" << sq_bits << ")";
	}

	virtual std::string get_tag() const {
		return "pq";
	}

	inline virtual int sq_dim() const {
		return dim/sq_count;
	}

	inline int sq_centroid_count() const {
		return 1 << sq_bits;
	}

	inline virtual int all_centroids_dim() const {
		// sq_dim * sq_count * sq_centroid_count
		// = dim/sq_count * sq_count * sq_centroid_count
		// = dim * sq_centroid_count
		return dim * sq_centroid_count();
	}

	inline void set_centroids(float** centroids_) {
		const int comp_count = all_centroids_dim();
		std::copy(centroids_[0], centroids_[0] + comp_count, centroids_flat.get());
	}

	inline void set_properties(int sq_count_, int sq_bits_, int dim_) {
		sq_count = sq_count_;
		sq_bits = sq_bits_;
		dim = dim_;
		setup_centroids();
	}

	inline void setup_centroids() {
		const int comp_count = all_centroids_dim();
		centroids_flat.reset(new float[comp_count]);
		centroids.reset(new float*[sq_count]);
		const int sq_centroids_dim = sq_centroid_count() * sq_dim();
		for (int sq_i = 0; sq_i < sq_count; ++sq_i) {
			centroids[sq_i] = centroids_flat.get() + sq_i * sq_centroids_dim;
		}
	}

	template<typename Archive>
	inline void save(Archive& ar) const {
		ar(sq_count, sq_bits, dim);
		std::size_t comp_count = all_centroids_dim();
		ar(
				cereal::binary_data(centroids_flat.get(),
						comp_count * sizeof(*centroids_flat.get())));
	}

	template<typename Archive>
	inline void load(Archive& ar) {
		ar(sq_count, sq_bits, dim);
		std::size_t comp_count = all_centroids_dim();
		setup_centroids();
		ar(
				cereal::binary_data(centroids_flat.get(),
						comp_count * sizeof(*centroids_flat.get())));
	}

	virtual void rotate_vector(float* vector) {

	}

	virtual void rotate_multiple_vectors(float* vector, int count) {

	}

	int code_size() {
		return sq_count * sq_bits / 8 ;
	}

	void encode_vector(float* vector, std::uint8_t* code) {
		rotate_vector(vector);
		int code_offset = 0;
        const int subq_dim = sq_dim();
		std::fill(code, code + code_size(), 0);
		for (int sq_i = 0; sq_i < sq_count; ++sq_i) {
			// Extract subvector
			float* subvector = subv(vector, dim, sq_i);
			// Assign to centroids
			int index_tmp;
            const int k = 1;
            find_k_neighbors(1, sq_centroid_count(), subq_dim, k, subvector,
                             centroids[sq_i], &index_tmp);
			assert(index_tmp > 0);
			// Pack into code
			set_bits_generic(code, static_cast<std::uint32_t>(index_tmp),
					code_offset);
			code_offset += sq_bits;
		}
	}

	void encode_multiple_vectors(float* vectors, std::uint8_t* codes,
			int count) {
		rotate_multiple_vectors(vectors, count);

		const int subq_dim = sq_dim();
		std::unique_ptr<float[]> subvectors = std::make_unique<float[]>(
				count * subq_dim);
		std::unique_ptr<int[]> assign = std::make_unique<int[]>(count);

		decltype(&multiple_set_bits_4) m_set_bits = prepare_multiple_set_bits(
				sq_bits, codes, count * code_size());

		for (int sq_i = 0; sq_i < sq_count; ++sq_i) {
			// Extract subvectors
			extract_subvectors(vectors, dim, count, subq_dim, sq_i,
					subvectors.get());
			// Assign to centroids
            const int k = 1;
            find_k_neighbors(count, sq_centroid_count(), subq_dim, k,
                             subvectors.get(), centroids[sq_i], assign.get());
			// Pack into codes
			m_set_bits(assign.get(), count, sq_count, sq_bits, sq_i, codes);
		}
	}
};

struct opq : public base_pq {
	std::unique_ptr<float[]> rotation;

	opq() {};

	opq(int sq_count_, int sq_bits_, int dim_, float* centroids_flat_ = nullptr,
			float* rotation_ = nullptr) :
			base_pq(sq_count_, sq_bits_, dim_, centroids_flat_) {
		setup_rotation();
		if (rotation_ != nullptr) {
			set_rotation(rotation_);
		}
	}

	virtual void print(std::ostream& os) const {
		os << get_tag() << " (dim=" << dim << ", sq=" << sq_count << "x" << sq_bits << ")";
	}

	virtual std::string get_tag() const {
		return "opq";
	}

	inline void setup_rotation() {
		rotation.reset(new float[dim*dim]);
	}

	inline void set_rotation(float* rotation_) {
		const int rotate_dim = dim * dim;
		std::copy(rotation_, rotation_ + rotate_dim, rotation.get());
	}

	virtual void rotate_vector(float* vector) {
		const int inc = 1;
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, alpha,
				rotation.get(), dim, vector, inc, beta, vector, inc);
		// You really really really need to verify this function
		assert(false);
	}

	virtual void rotate_multiple_vectors(float* vectors, int count) {
		const float alpha = 1.0f;
		const float beta = 0.0f;

		// Safe with copy
		std::unique_ptr<float[]> rotated_vectors = std::make_unique<float[]>(
				count * dim);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, dim, dim,
				alpha, vectors, dim, rotation.get(), dim, beta,
				rotated_vectors.get(), dim);
		std::copy(rotated_vectors.get(), rotated_vectors.get() + count * dim,
				vectors);
	}

	template<typename Archive>
	inline void save(Archive& ar) const {
		// Save PQ
		ar(cereal::base_class<base_pq>(this));
		// Save rotation matrix
		ar(
				cereal::binary_data(rotation.get(),
						dim * dim * sizeof(*rotation.get())));
	}

	template<typename Archive>
	inline void load(Archive& ar) {
		// Load PQ
		ar(cereal::base_class<base_pq>(this));
		// Load rotation matrix
		setup_rotation();
		ar(
				cereal::binary_data(rotation.get(),
						dim * dim * sizeof(*rotation.get())));

	}
};

std::ostream& operator<<(std::ostream& os, const base_pq& pq);

#include <cereal/archives/binary.hpp>
CEREAL_REGISTER_TYPE(opq);
CEREAL_REGISTER_POLYMORPHIC_RELATION(base_pq, opq);

std::unique_ptr<base_pq> pq_from_data_file(const char* data_filename);

#endif /* QUANTIZERS_HPP_ */
