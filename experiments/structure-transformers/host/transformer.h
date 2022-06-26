#pragma once
#include <noarr/structures.hpp>

using dim_t = char;

struct base_host_transformer_2d
{
public:
	template <dim_t X, dim_t Y, typename bag_in_t, typename bag_out_t>
	static void transform(const bag_in_t& input_bag, bag_out_t& output_bag)
	{
		const size_t X_size = input_bag.structure().template get_length<X>();
		const size_t Y_size = input_bag.structure().template get_length<Y>();

		for (size_t i = 0; i < X_size; i++)
			for (size_t j = 0; j < Y_size; j++)
				output_bag.template at<X, Y>(i, j) = input_bag.template at<X, Y>(i, j);
	}
};
