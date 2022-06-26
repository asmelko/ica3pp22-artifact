#pragma once

#include <map>
#include <vector>

enum class memory_type_t
{
	CUDA_GLOBAL,
	CUDA_SHARED,
	HOST
};


enum class access_type_t
{
	READ,
	WRITE
};

struct trace_t
{
	memory_type_t memory_type;
	access_type_t access_type;
	size_t worker_id;
	uintptr_t address;
	size_t size;

	constexpr trace_t(memory_type_t memory_type, access_type_t access_type, size_t worker_id, uintptr_t address,
					  size_t size)
		: memory_type(memory_type), access_type(access_type), worker_id(worker_id), address(address), size(size)
	{}

	trace_t() = default;

	bool operator==(const trace_t& oth) const
	{
		return std::tie(access_type, address, memory_type, size, worker_id)
			   == std::tie(oth.access_type, oth.address, oth.memory_type, oth.size, oth.worker_id);
	}
};

using worker_trace_map_t = std::map<size_t, std::vector<trace_t>>;
