#pragma once

#include "memory-traces/trace.h"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/structs.hpp"

template <typename Structure, typename BagPolicy>
class trace_bag : public noarr::bag<Structure, BagPolicy>
{
	memory_type_t memory_type_ = memory_type_t::HOST;
	mutable worker_trace_map_t traces_;

	template <typename T>
	void add_trace(access_type_t access_type, size_t worker_id, T* address) const
	{
		traces_[worker_id].emplace_back(memory_type_, access_type, worker_id, (uintptr_t)address,
										sizeof(noarr::scalar_t<Structure>));
	}

public:
	using base = noarr::bag<Structure, BagPolicy>;

	using base::bag;

	void set_memory_type(memory_type_t memory_type) { memory_type_ = memory_type; }

	const worker_trace_map_t& get_traces() const { return traces_; }

	template <char... Dims, typename... Ts>
	decltype(auto) at(size_t worker_id, Ts... ts)
	{
		auto&& address = base::template at<Dims...>(ts...);
		add_trace(access_type_t::WRITE, worker_id, &address);
		return address;
	}

	template <char... Dims, typename... Ts>
	decltype(auto) at(size_t worker_id, access_type_t access_type, Ts... ts)
	{
		auto&& address = base::template at<Dims...>(ts...);
		add_trace(access_type, worker_id, &address);
		return address;
	}

	template <char... Dims, typename... Ts>
	decltype(auto) at(size_t worker_id, Ts... ts) const
	{
		auto&& address = base::template at<Dims...>(ts...);
		add_trace(access_type_t::READ, worker_id, &address);
		return address;
	}
};

template <typename Structure>
constexpr auto make_trace_bag(Structure s, memory_type_t memory_type = memory_type_t::HOST)
{
	auto bag = trace_bag<Structure, noarr::helpers::bag_policy<std::unique_ptr>>(s);
	bag.set_memory_type(memory_type);
	return bag;
}
