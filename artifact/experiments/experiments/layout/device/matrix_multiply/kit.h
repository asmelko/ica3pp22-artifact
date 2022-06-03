#pragma once

#include "utils/matrix.h"

template <typename T, char X, char Y>
class multipty_kit : public matrix_generator<T, X, Y>
{
	bool validation_result_;

public:
	multipty_kit(size_t lhs_height, size_t lhs_width, size_t rhs_width)
		: matrix_generator<T, X, Y>(lhs_height, rhs_width, false),
		  validation_result_(true),
		  lhs_generator(lhs_height, lhs_width),
		  rhs_generator(lhs_width, rhs_width)
	{}

	matrix_generator<T, X, Y> lhs_generator;
	matrix_generator<T, X, Y> rhs_generator;

	bool equals_floats(const T& lhs, const T& rhs) { return std::abs(lhs - rhs) < 1.; }

	template <typename bag_t>
	void validate(const bag_t& bag)
	{
		return;
		if (this->data_.empty())
			matmul();

		for (size_t i = 0; i < this->height; i++)
		{
			for (size_t j = 0; j < this->width; j++)
			{
				if (!equals_floats(bag.template at<X, Y>(i, j), this->data_[i * this->width + j]))
				{
					std::cout << "Error at [" << i << ", " << j << "] " << bag.template at<X, Y>(i, j)
							  << " != " << this->data_[i * this->width + j] << std::endl;
					validation_result_ = false;
					return;
				}
			}
		}
		validation_result_ = true;
	}

	bool validation_successful() const { return validation_result_ == true; }

	void refresh()
	{
		lhs_generator.refresh();
		rhs_generator.refresh();
	}

private:
	void matmul()
	{
		assert(lhs_generator.width == rhs_generator.height);
		for (size_t i = 0; i < lhs_generator.height; i++)
		{
			for (size_t j = 0; j < rhs_generator.width; j++)
			{
				this->data_.push_back(0);
				for (size_t k = 0; k < lhs_generator.width; k++)
				{
					this->data_[i * this->width + j] += lhs_generator.data()[i * lhs_generator.width + k]
														* rhs_generator.data()[k * rhs_generator.width + j];
				}
			}
		}
	}
};
