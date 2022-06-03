#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <chrono>
#include <vector>

class experiment
{
private:
	void warm_up()
	{
		(void)run_base();
		auto start = std::chrono::steady_clock::now();
		std::chrono::duration<double> warmup_duration;
		do
		{
			(void)run_base();
			warmup_duration = std::chrono::steady_clock::now() - start;
		} while (warmup_duration.count() < 5.0);
	}

protected:
	struct result
	{
		std::string name;
		double transform_duration, action_duration, overall_duration;

		result(std::string&& name, std::tuple<double, double, double> durations)
			: name(std::move(name)),
			  transform_duration(std::get<0>(durations)),
			  action_duration(std::get<1>(durations)),
			  overall_duration(std::get<2>(durations))
		{
			if (this->name.back() == ' ')
				this->name.pop_back();
		}

		void print(std::ostream& out) const
		{
			print_table();
			std::cout << std::setw(30) << "Structures" << std::setw(12) << "Transform t" << std::setw(12) << "Action t"
					  << std::setw(12) << "Overall t" << std::endl;
			std::cout << std::setw(30) << name << std::setw(12) << std::to_string(transform_duration) + "s"
					  << std::setw(12) << std::to_string(action_duration) + "s" << std::setw(12)
					  << std::to_string(overall_duration) + "s" << std::endl;
			print_table();
		}

		bool operator<(const result& oth) const
		{
			return std::tie(action_duration, transform_duration, overall_duration)
				   < std::tie(oth.action_duration, oth.transform_duration, oth.overall_duration);
		}

	private:
		void print_table() const { std::cout << std::string(66, '-') << std::endl; }
	};

	virtual result run_base() = 0;
	virtual result run_optimized() = 0;
	virtual std::vector<result> run_experiments() = 0;

public:
	void run()
	{
		{
			std::cout << "Running experiments ..." << std::endl;
			auto results = run_experiments();
			std::sort(results.begin(), results.end());
			std::for_each(results.begin(), results.end(), [&](const auto& res) { res.print(std::cout); });
		}
	}
};
