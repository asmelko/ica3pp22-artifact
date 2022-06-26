#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "examples.h"
#include "noarr/structures_extended.hpp"
#include "utils/timer.h"

namespace {

constexpr size_t buffer_size_x = 1 << 20;
constexpr size_t buffer_size_y = 1 << 5;
constexpr size_t buffer_size_z = 1 << 5;

volatile size_t vsize_x = buffer_size_x;
volatile size_t vsize_y = buffer_size_y;
volatile size_t vsize_z = buffer_size_z;

size_t size_x = vsize_x;
size_t size_y = vsize_y;
size_t size_z = vsize_z;

using data_t = float;

using cube_t = noarr::array<'x', buffer_size_x,
							noarr::array<'y', buffer_size_y, noarr::array<'z', buffer_size_z, noarr::scalar<data_t>>>>;

} // namespace


template <typename bag_t>
inline void run(const bag_t& in, bag_t& out, const size_t x, const size_t y, const size_t z)
{
	data_t sum = in.template at<'x', 'y', 'z'>(x, y, z);
	sum += in.template at<'x', 'y', 'z'>(x + 1, y, z);
	sum += in.template at<'x', 'y', 'z'>(x - 1, y, z);
	sum += in.template at<'x', 'y', 'z'>(x, y + 1, z);
	sum += in.template at<'x', 'y', 'z'>(x, y - 1, z);
	sum += in.template at<'x', 'y', 'z'>(x, y, z + 1);
	sum += in.template at<'x', 'y', 'z'>(x, y, z - 1);
	out.template at<'x', 'y', 'z'>(x, y, z) = sum / 7;
}

template <typename F>
inline void run(const F* in, F* out, const size_t x, const size_t y, const size_t z)
{
	data_t sum = in[x * size_y * size_z + y * size_z + z];
	sum += in[(x + 1) * size_y * size_z + y * size_z + z];
	sum += in[(x - 1) * size_y * size_z + y * size_z + z];
	sum += in[x * size_y * size_z + (y + 1) * size_z + z];
	sum += in[x * size_y * size_z + (y - 1) * size_z + z];
	sum += in[x * size_y * size_z + y * size_z + z + 1];
	sum += in[x * size_y * size_z + y * size_z + z - 1];
	out[x * size_y * size_z + y * size_z + z] = sum / 7;
}

template <typename... Args>
void run_internal(Args&&... args)
{
	for (size_t x = 1; x < size_x - 1; x++)
		for (size_t y = 1; y < size_y - 1; y++)
			for (size_t z = 1; z < size_z - 1; z++)
				run(std::forward<Args>(args)..., x, y, z);
}

struct tag_t
{};

template <typename... Args>
constexpr void run_internal(const noarr::bag<Args...> in_bag, noarr::bag<Args...> out_bag, tag_t)
{
	for (size_t x = 1; x < in_bag.template get_length<'x'>() - 1; x++)
		for (size_t y = 1; y < in_bag.template get_length<'y'>() - 1; y++)
			for (size_t z = 1; z < in_bag.template get_length<'z'>() - 1; z++)
				run(in_bag, out_bag, x, y, z);
}

template <typename F>
void fill(F* data)
{
	std::mt19937 gen;
	std::uniform_real_distribution<double> distrib;

	for (size_t x = 0; x < size_x * size_y * size_z; x++)
	{
		data[x] = (F)distrib(gen);
	}
}

template <typename F>
void compare(const F* l, const F* r)
{
	bool is_err = false;
	for (size_t x = 0; x < size_x * size_y * size_z; x++)
	{
		if (std::abs(l[x] - r[x]) > 1.0)
		{
			std::cerr << l[x] << "!=" << r[x] << std::endl;
			is_err = true;
		}
	}

	if (is_err)
		std::cerr << "FAIL!" << std::endl;
}

void run_once(bool verify, size_t iters, bool constant_loop)
{
	std::vector<data_t> tmp;
	tmp.resize(size_x * size_y * size_z);
	auto out_bag = noarr::make_bag(cube_t(), (char*)tmp.data());
	{
		std::vector<data_t> in_vec;
		in_vec.resize(size_x * size_y * size_z);
		fill(in_vec.data());

		auto in_bag = noarr::make_bag(cube_t(), (char*)in_vec.data());

		std::chrono::duration<double> duration;

		for (size_t i = 0; i < iters; i++)
		{
			if (constant_loop)
			{
				timer t(duration);

				run_internal(in_bag, out_bag, tag_t {});
			}
			else
			{
				timer t(duration);

				run_internal(in_bag, out_bag);
			}

			std::cout << "static," << duration.count() << std::endl;

			std::swap(in_bag, out_bag);
			
			// notification of activity
			std::cerr << ".";
			std::cerr.flush();
		}
		std::cerr << std::endl;
	}

	std::vector<data_t> out_vec;
	{
		std::vector<data_t> in_vec;
		in_vec.resize(size_x * size_y * size_z);
		fill(in_vec.data());

		out_vec.resize(size_x * size_y * size_z);

		std::chrono::duration<double> duration;

		for (size_t i = 0; i < iters; i++)
		{
			{
				timer t(duration);

				run_internal(in_vec.data(), out_vec.data());
			}

			std::cout << "dynamic," << duration.count() << std::endl;

			std::swap(in_vec, out_vec);
			std::cerr << ".";
			std::cerr.flush();
		}
		std::cerr << std::endl;
	}

	if (verify)
		compare((data_t*)out_bag.data(), out_vec.data());
}

void run_stencil(const std::vector<std::string>& args)
{
	size_t reps = args.size() != 0 && args.back() == "--quick" ? 5 : 100;
	std::cerr << "Running " << reps << " repetitions:" << std::endl;

	std::cout << "alg,time" << std::endl;

	bool const_loop = args.size() != 0 && args[0] == "cloop";

	run_once(true, reps, const_loop);
}
