#pragma once

#include "utils/matrix.h"

template <typename T, char X, char Y>
class transpose_kit : public matrix_generator<T, X, Y>
{
	bool validation_result_;

public:
	transpose_kit(size_t height, size_t width) : matrix_generator<T, X, Y>(height, width), validation_result_(true) {}

	template <typename bag_t>
	void validate(const bag_t& bag)
	{
		for (size_t i = 0; i < this->height; i++)
		{
			for (size_t j = 0; j < this->width; j++)
			{
				if (bag.template at<X, Y>(j, i) != this->data_[i * this->width + j])
				{
					std::cout << "Error at [" << i << ", " << j << "] " << bag.template at<X, Y>(i, j)
							  << " != " << this->data_[i * this->width + j] << std::endl;
					validation_result_ = false;
					return;
				}
			}
		}
		validation_result_ = true;
		return;
	}

	bool validation_successful() const { return validation_result_ == true; }
};
