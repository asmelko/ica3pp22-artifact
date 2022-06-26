#include <cmath>
#include <iostream>
#include <random>

#include "examples.h"
#include "utils/timer.h"
using data_t = float;

constexpr size_t kk = 4;

auto query(const data_t* ref, const data_t* points, size_t n, size_t d)
{
	size_t closest[kk] = { 0, 0, 0, 0 };
	data_t min[kk] = { std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max(),
					   std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max() };


	for (size_t i = 0; i < n; i++)
	{
		for (size_t k = 0; k < kk; k++)
		{
			data_t sum = .0f;

			for (size_t j = 0; j < d; j++)
			{
				const auto tmp = ref[k * d + j] - points[i * d + j];
				sum += tmp * tmp;
			}
			sum = std::sqrt(sum);

			if (sum < min[k])
			{
				min[k] = sum;
				closest[k] = i;
			}
		}
	}

	return std::make_tuple(closest[0], closest[1], closest[2], closest[3]);
}

template <size_t d>
auto query(const data_t* ref, const data_t* points, size_t n)
{
	size_t closest[kk] = { 0, 0, 0, 0 };
	data_t min[kk] = { std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max(),
					   std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max() };


	for (size_t i = 0; i < n; i++)
	{
		for (size_t k = 0; k < kk; k++)
		{
			data_t sum = .0f;

			for (size_t j = 0; j < d; j++)
			{
				const auto tmp = ref[k * d + j] - points[i * d + j];
				sum += tmp * tmp;
			}
			sum = std::sqrt(sum);

			if (sum < min[k])
			{
				min[k] = sum;
				closest[k] = i;
			}
		}
	}

	return std::make_tuple(closest[0], closest[1], closest[2], closest[3]);
}

template <size_t d>
struct s
{
	data_t data[d];
};

template <size_t d>
auto query(const s<d>* ref, const s<d>* points, size_t n)
{
	size_t closest[kk] = { 0, 0, 0, 0 };
	data_t min[kk] = { std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max(),
					   std::numeric_limits<data_t>::max(), std::numeric_limits<data_t>::max() };


	for (size_t i = 0; i < n; i++)
	{
		for (size_t k = 0; k < kk; k++)
		{
			data_t sum = .0f;

			for (size_t j = 0; j < d; j++)
			{
				const auto tmp = ref[k].data[j] - points[i].data[j];
				sum += tmp * tmp;
			}
			sum = std::sqrt(sum);

			if (sum < min[k])
			{
				min[k] = sum;
				closest[k] = i;
			}
		}
	}

	return std::make_tuple(closest[0], closest[1], closest[2], closest[3]);
}

void run_knn(const std::vector<std::string>& args)
{
	size_t d = std::stoi(args[0]);

	size_t n = (1 << 20) * 10;
	std::vector<data_t> data;
	data.resize(n * d);

	std::random_device rd_;
	std::mt19937 gen_(rd_());
	std::uniform_real_distribution<data_t> distrib_;
	for (size_t i = 0; i < n * d; i++)
	{
		data[i] = distrib_(gen_);
	}

	for (size_t i = 0; i < 10; i++)
	{
		{
			std::chrono::duration<double> duration;

			{
				timer t(duration);

				query(data.data() + (n - kk) * d, data.data(), n, d);
			}

			std::cout << "dynamic," << duration.count() << std::endl;
		}

		{
			std::chrono::duration<double> duration;

			{
				timer t(duration);

				if (d == 2)
					query<2>(data.data() + (n - kk) * 2, data.data(), n);
				if (d == 4)
					query<4>(data.data() + (n - kk) * 4, data.data(), n);
				else if (d == 8)
					query<8>(data.data() + (n - kk) * 8, data.data(), n);
				else if (d == 16)
					query<16>(data.data() + (n - kk) * 16, data.data(), n);
				else if (d == 32)
					query<32>(data.data() + (n - kk) * 32, data.data(), n);
			}

			std::cout << "static," << duration.count() << std::endl;
		}

		{
			std::chrono::duration<double> duration;

			{
				timer t(duration);

				if (d == 2)
				{
					constexpr size_t dd = 2;
					auto sdata = (s<dd>*)data.data();
					query<dd>(sdata + (n - kk), sdata, n);
				}
				if (d == 4)
				{
					constexpr size_t dd = 4;
					auto sdata = (s<dd>*)data.data();
					query<dd>(sdata + (n - kk), sdata, n);
				}
				else if (d == 8)
				{
					constexpr size_t dd = 8;
					auto sdata = (s<dd>*)data.data();
					query<dd>(sdata + (n - kk), sdata, n);
				}
				else if (d == 16)
				{
					constexpr size_t dd = 16;
					auto sdata = (s<dd>*)data.data();
					query<dd>(sdata + (n - kk), sdata, n);
				}
				else if (d == 32)
				{
					constexpr size_t dd = 32;
					auto sdata = (s<dd>*)data.data();
					query<dd>(sdata + (n - kk), sdata, n);
				}
			}

			std::cout << "sstatic," << duration.count() << std::endl;
		}
	}
}