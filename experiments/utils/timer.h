#pragma once

#include <chrono>
#include <memory>
#include <vector>

struct timer
{
private:
	std::chrono::time_point<std::chrono::steady_clock> start_;
	std::chrono::duration<double>& elapsed_;

public:
	timer(std::chrono::duration<double>& duration) : start_(std::chrono::steady_clock::now()), elapsed_(duration) {}

	~timer() { elapsed_ = std::chrono::steady_clock::now() - start_; }
};

using duration_ptr = std::shared_ptr<std::chrono::duration<double>>;

struct duration_composite
{
	std::vector<duration_ptr> composite;

	double count()
	{
		double count = 0;
		for (const auto& duration : composite)
			count += duration->count();
		return count;
	}

	void add(duration_ptr duration) { composite.push_back(duration); }

	void clear() { composite.clear(); }
};
