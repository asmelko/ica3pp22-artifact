#pragma once

#include <cassert>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "../trace.h"

class cuda_memory_model
{
	static const size_t global_request_size = 128;
	static const size_t shared_request_size = 4;
	static const size_t warp_size = 32;

	using trace_iterator = worker_trace_map_t::const_iterator;

	using range_t = std::pair<uintptr_t, uintptr_t>;

	struct score_t
	{
		size_t score;
		memory_type_t memory_type;
	};

	bool check_step_(trace_iterator begin, trace_iterator end)
	{
		std::advance(begin, warp_size);
		return begin == end;
	}

	size_t score_global_access(const trace_iterator& begin, const trace_iterator& end, size_t step_id)
	{
		assert(check_step_(begin, end));

		std::set<range_t> requests;
		for (auto it = begin; it != end; ++it)
		{
			auto&& trace = it->second[step_id];

			requests.emplace(trace.address, trace.address + trace.size - 1);
		}

		assert(requests.size() >= 1);

		size_t coalesced = 1;

		size_t sector = requests.begin()->first / global_request_size;

		auto check_sector = [&](uintptr_t address) {
			size_t this_sector = address / global_request_size;
			if (this_sector > sector)
			{
				sector = this_sector;
				coalesced++;
			}
		};

		for (auto it = requests.begin(); it != requests.end(); ++it)
		{
			check_sector(it->first);
			check_sector(it->second);
		}

		assert(coalesced >= 1);

		return coalesced;
	}

	size_t score_shared_access(const trace_iterator& begin, const trace_iterator& end, size_t step_id)
	{
		assert(check_step_(begin, end));

		std::set<range_t> requests;
		for (auto it = begin; it != end; ++it)
		{
			auto&& trace = it->second[step_id];

			requests.emplace(trace.address, trace.address + trace.size - 1);
		}

		std::map<uintptr_t, size_t> used_banks;

		size_t max_conflict = 0;

		for (const auto& req : requests)
		{
			size_t bank = req.first % shared_request_size;
			for (uintptr_t i = req.first; i <= req.second; i++)
			{
				auto current_bank = i % shared_request_size;
				if (i == req.first || current_bank != bank)
				{
					bank = current_bank;
					if (auto it = used_banks.find(bank); it != used_banks.end())
					{
						used_banks[bank]++;
						if (max_conflict < used_banks[bank])
							max_conflict = used_banks[bank];
					}
					else
						used_banks[bank] = 1;
				}
			}
		}

		assert(max_conflict != 1 && max_conflict <= warp_size);

		return max_conflict;
	}

	score_t score_step(const trace_iterator& begin, const trace_iterator& end, size_t step_id)
	{
		switch (begin->second[step_id].memory_type)
		{
			case memory_type_t::CUDA_GLOBAL:
				return score_t { score_global_access(begin, end, step_id), memory_type_t::CUDA_GLOBAL };
			case memory_type_t::CUDA_SHARED:
				return score_t { score_shared_access(begin, end, step_id), memory_type_t::CUDA_SHARED };
			default:
				exit(0);
				return score_t { 0, memory_type_t::CUDA_SHARED };
		}
	}

	bool check_length_(const trace_iterator& begin, const trace_iterator& end)
	{
		size_t length = begin->second.size();
		for (auto it = begin; it != end; ++it)
			if (length != it->second.size())
				return false;
		return true;
	}

	double score_warp(const trace_iterator& begin, const trace_iterator& end)
	{
		assert(check_length_(begin, end));

		std::vector<double> scores;
		for (size_t i = 0; i < begin->second.size(); i++)
		{
			auto score = score_step(begin, end, i);

			if (score.memory_type == memory_type_t::CUDA_GLOBAL)
			{
				auto tmp = std::log2(score.score) / std::log2(32);
				tmp = std::min(tmp, 1.0);

				scores.push_back(tmp);
			}
			else
			{
				auto tmp = std::log1p(score.score) / std::log1p(32);
				tmp = std::min(tmp, 1.0);

				scores.push_back(tmp);
			}
		}

		double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
		double mean = sum / scores.size();

		return mean;
	}

public:
	std::vector<double> score(const worker_trace_map_t& trace_map)
	{
		if (trace_map.size() % warp_size != 0)
			exit(0);

		std::vector<double> warp_scores;

		for (auto it = trace_map.begin(); it != trace_map.end(); std::advance(it, warp_size))
		{
			auto end = it;
			std::advance(end, warp_size);
			warp_scores.push_back(score_warp(it, end));
		}

		return warp_scores;
	}
};
