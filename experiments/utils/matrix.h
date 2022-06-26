#pragma once

#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

template <typename T, char X, char Y>
class matrix_generator
{
	std::random_device rd_;
	std::mt19937 gen_;
	std::uniform_real_distribution<double> distrib_;

protected:
	std::vector<T> data_;
	bool generated_;

	matrix_generator(size_t height, size_t width, bool initialize)
		: gen_(rd_()), distrib_(-100, 100), generated_(false), height(height), width(width)
	{
		if (!initialize)
			return;

		data_.resize(width * height);
		for (size_t i = 0; i < height* width; i++)
		{
			data_[i] = (T)distrib_(gen_);
		}
	}

public:
	size_t height, width;

	matrix_generator(size_t height, size_t width) : matrix_generator(height, width, true) {}

	T* data() { return data_.data(); }

	const T* data() const { return data_.data(); }

	template <typename bag_t>
	void generate(bag_t& bag)
	{
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
			{
				bag.template at<X, Y>(i, j) = data_[i * width + j];
			}
		}

		generated_ = true;
	}

	bool generating_finished() const { return generated_; }

	void refresh() { generated_ = false; }
};
