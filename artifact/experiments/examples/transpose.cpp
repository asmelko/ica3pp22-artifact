#include <iostream>
#include <string>

#include "layouts/z_curve.h"
#include "noarr/structures_extended.hpp"
#include "utils/timer.h"

using matrix_zcurve = noarr::z_curve<'n', 'm', noarr::sized_vector<'a', noarr::scalar<float>>>;


template <typename bag_t>
void transpose(bag_t& bag, size_t N)
{
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = i + 1; j < N; ++j)
		{
			std::swap(bag.template at<'m', 'n'>(i, j), bag.template at<'m', 'n'>(j, i));
		}
	}
}

template <typename bag_t>
void fill(bag_t& bag, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			bag.template at<'m', 'n'>(i, j) = (float)(i * N + j);
		}
	}
}

template <typename bag_1, typename bag_2>
void compare(const bag_1& bag1, const bag_2& bag2, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (std::abs(bag1.template at<'m', 'n'>(i, j) - bag2.template at<'m', 'n'>(i, j)) > 1)
				std::cout << bag1.template at<'m', 'n'>(i, j) << " != " << bag2.template at<'m', 'n'>(i, j)
						  << std::endl;
		}
	}
}

void run_transpose(const std::vector<std::string>& args)
{
	auto size = (size_t)std::stoi(args[0]);


	auto s1 = matrix_zcurve(noarr::sized_vector<'a', noarr::scalar<float>>(noarr::scalar<float>(), size * size),
							noarr::helpers::z_curve_bottom<'n'>(size), noarr::helpers::z_curve_bottom<'m'>(size));

	auto bag1 = noarr::make_bag(s1);
	fill(bag1, size);

	{
		std::chrono::duration<double> duration;
		{
			timer t(duration);
			transpose(bag1, size);
		}
		std::cout << "z " << duration.count() << std::endl;
	}



	auto s2 = noarr::vector<'m', noarr::vector<'n', noarr::scalar<float>>>() | noarr::set_length<'m'>(size)
			  | noarr::set_length<'n'>(size);

	auto bag2 = noarr::make_bag(s2);
	fill(bag2, size);

	{
		std::chrono::duration<double> duration;
		{
			timer t(duration);
			transpose(bag2, size);
		}
		std::cout << "rowcol " << duration.count() << std::endl;
	}


	compare(bag1, bag2, size);
}