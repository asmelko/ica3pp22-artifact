#pragma once

#include <utility>

#include "experiments/layout/experiment.h"
#include "layouts/layouts.h"
#include "structure-transformers/device/transformer_pipeline.cuh"
#include "utils/pipelines.h"

template <class derived>
class device_experiment : public experiment
{
	size_t experiments_repetitions_ = 20;

protected:
	using sizes_t = std::pair<size_t, size_t>;
	using data_t = float;

	using matrix_rows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<data_t>>>;
	using matrix_cols = noarr::vector<'n', noarr::vector<'m', noarr::scalar<data_t>>>;

	duration_composite transform_duration, action_duration;

	template <typename... args_t>
	std::tuple<double, double, double> run_single(bool optimized, const std::vector<std::string>& structs = {})
	{
		double transform_acc = 0;
		double action_acc = 0;
		double overall_acc = 0;
		for (size_t i = 0; i < experiments_repetitions_; i++)
		{
			auto start_overall = std::chrono::steady_clock::now();
			{
				smart_ptr_scheduler scheduler;
				static_cast<derived*>(this)->template construct<args_t...>(scheduler, optimized);
				scheduler.run();
			}
			std::chrono::duration<double> duration_overall = std::chrono::steady_clock::now() - start_overall;

			for (const auto& s : structs)
				std::cerr << s << ",";

			std::cerr << transform_duration.count() << "," << action_duration.count() << "," << duration_overall.count()
					  << std::endl;

			transform_acc += transform_duration.count();
			action_acc += action_duration.count();
			overall_acc += duration_overall.count();

			static_cast<derived*>(this)->kit.refresh();
			action_duration.clear();
			transform_duration.clear();
		}

		return std::make_tuple(transform_acc / experiments_repetitions_, action_acc / experiments_repetitions_,
							   overall_acc / experiments_repetitions_);
	}

	template <typename transture_t, typename scheduler_t, typename generator_t>
	auto construct_in_tr_generic(scheduler_t& scheduler, sizes_t sizes, generator_t& generator)
	{
		size_t buffer_size = sizes.first * sizes.second * sizeof(data_t);
		auto rows_structure =
			matrix_rows() | noarr::set_length<'m'>(sizes.first) | noarr::set_length<'n'>(sizes.second);
		auto [in_hub, in_node] = construct_generator(generator, rows_structure, buffer_size);

		if constexpr (!std::is_same_v<decltype(rows_structure), transture_t>)
		{
			auto [tr_hub, tr_node] = construct_cuda_transformer<'m', 'n', transture_t>(*in_hub, buffer_size);

			transform_duration.add(tr_node->advance_duration);
			auto* tmp = tr_hub.get();
			scheduler << std::move(tr_hub) << std::move(tr_node) << std::move(in_hub) << std::move(in_node);
			return tmp;
		}
		else
		{
			auto* tmp = in_hub.get();
			scheduler << std::move(in_hub) << std::move(in_node);
			return tmp;
		}
	}

	template <typename transture_t, typename scheduler_t, typename validator_t>
	auto construct_out_tr_generic(scheduler_t& scheduler, sizes_t sizes, validator_t& validator)
	{
		size_t buffer_size = sizes.first * sizes.second * sizeof(data_t);

		auto rows_structure =
			matrix_rows() | noarr::set_length<'m'>(sizes.first) | noarr::set_length<'n'>(sizes.second);

		if constexpr (!std::is_same_v<decltype(rows_structure), transture_t>)
		{
			auto mid_hub = construct_hub<transture_t>(buffer_size);

			auto [tr_hub, tr_node] =
				construct_cuda_transformer<'m', 'n', decltype(rows_structure)>(*mid_hub, buffer_size);

			transform_duration.add(tr_node->advance_duration);

			auto out_node = construct_validator(validator, rows_structure, *tr_hub);

			auto* tmp = mid_hub.get();
			scheduler << std::move(tr_hub) << std::move(tr_node) << std::move(mid_hub) << std::move(out_node);
			return tmp;
		}
		else
		{
			auto [out_hub, out_node] = construct_validator(validator, rows_structure, buffer_size);

			auto* tmp = out_hub.get();
			scheduler << std::move(out_hub) << std::move(out_node);
			return tmp;
		}
	}

	template <class... T>
	std::vector<result> compose(T... results)
	{
		std::vector<result> res;

		if constexpr (std::is_same_v<std::tuple_element_t<0, std::tuple<T...>>, result>)
			(res.push_back(results), ...);
		else
			(res.insert(res.end(), results.begin(), results.end()), ...);

		return res;
	}

	template <size_t N, class tuple_t, size_t... I, typename... args_t>
	auto select(const tuple_t& tuple, std::index_sequence<I...> idxs, args_t... args)
	{
		if constexpr (N == 0)
		{
			std::cout << ".";
			std::cout.flush();
			std::vector<std::string> structs;
			(structs.push_back(args.first), ...);
			return result((args.first + ... + ""), run_single<typename args_t::second_type...>(false, structs));
		}
		else
			return compose(select<N - 1>(tuple, idxs, args..., std::get<I>(tuple))...);
	}

	virtual std::vector<result> run_experiments() override
	{
		for (size_t i = 0; i < derived::number_args; i++)
			std::cerr << "s" << i << ",";
		std::cerr << "transform,action,overall" << std::endl;

		auto types = static_cast<derived*>(this)->types;
		constexpr size_t transture_count = std::tuple_size_v<decltype(types)>;

		auto ret = select<derived::number_args>(types, std::make_index_sequence<transture_count>());
		std::cout << std::endl;
		return ret;
	}
};
