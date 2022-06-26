#include <gtest/gtest.h>
#include <noarr/structures.hpp>
#include <noarr/structures_extended.hpp>

#include "memory-traces/host/trace_bag.h"

TEST(trace, bag)
{
	auto structure = noarr::sized_vector<'a', noarr::scalar<int>>(noarr::scalar<int>(), 10);

	auto bag = make_trace_bag(structure);

	bag.at<'a'>(1, access_type_t::READ, 1) = 1;
	bag.at<'a'>(0, access_type_t::WRITE, 0) = 1;

	auto&& map = bag.get_traces();

	ASSERT_EQ(map.size(), 2);

	{
		auto&& traces = map.at(0);
		ASSERT_EQ(traces.size(), 1);
		ASSERT_EQ(traces.front(),
				  trace_t(memory_type_t::HOST, access_type_t::WRITE, 0, (uintptr_t)bag.data(), sizeof(int)));
	}

	{
		auto&& traces = map.at(1);
		ASSERT_EQ(traces.size(), 1);
		ASSERT_EQ(traces.front(), trace_t(memory_type_t::HOST, access_type_t::READ, 1,
										  (uintptr_t)bag.data() + sizeof(int), sizeof(int)));
	}
}

TEST(trace, const_bag)
{
	auto structure = noarr::sized_vector<'a', noarr::scalar<int>>(noarr::scalar<int>(), 10);

	const auto bag = make_trace_bag(structure, memory_type_t::CUDA_GLOBAL);

	(void)bag.at<'a'>(1, 1);
	(void)bag.at<'a'>(0, 0);

	auto&& map = bag.get_traces();

	ASSERT_EQ(map.size(), 2);

	{
		auto&& traces = map.at(0);
		ASSERT_EQ(traces.size(), 1);
		ASSERT_EQ(traces.front(),
				  trace_t(memory_type_t::CUDA_GLOBAL, access_type_t::READ, 0, (uintptr_t)bag.data(), sizeof(int)));
	}

	{
		auto&& traces = map.at(1);
		ASSERT_EQ(traces.size(), 1);
		ASSERT_EQ(traces.front(), trace_t(memory_type_t::CUDA_GLOBAL, access_type_t::READ, 1,
										  (uintptr_t)bag.data() + sizeof(int), sizeof(int)));
	}
}