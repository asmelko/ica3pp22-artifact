#pragma once
#include <memory>

#include <noarr/cuda-pipelines.hpp>
#include <noarr/pipelines.hpp>
#include <noarr/structures-pipelines.hpp>

#include "timer.h"

template <typename structure_t>
std::unique_ptr<noarr::pipelines::Hub<structure_t>> construct_hub(size_t buffer_size)
{
	auto hub = std::make_unique<noarr::pipelines::Hub<structure_t>>(buffer_size);
	hub->allocate_envelopes(noarr::pipelines::Device::HOST_INDEX, 1);
	hub->allocate_envelopes(noarr::pipelines::Device::DEVICE_INDEX, 1);

	return hub;
}

template <typename generator_t, typename structure_t>
auto construct_generator(generator_t& generator, const structure_t& structure, size_t buffer_size)
{
	auto generator_hub = construct_hub<structure_t>(buffer_size);
	auto generator_node = std::make_unique<noarr::pipelines::LambdaComputeNode>("generate");

	auto& link = generator_node->link(generator_hub->to_produce(noarr::pipelines::Device::HOST_INDEX));

	generator_node->can_advance([&]() { return !generator.generating_finished(); });

	generator_node->advance([&, &generator_node = *generator_node, structure]() {
		link.envelope->structure = structure;
		auto bag = noarr::pipelines::bag_from_link(link);

		generator.generate(bag);

		generator_node.callback();
	});

	return std::make_pair(std::move(generator_hub), std::move(generator_node));
}

template <typename validator_t, typename structure_t, typename in_hub_t>
auto construct_validator(validator_t& validator, const structure_t& structure, in_hub_t& input_hub)
{
	auto validator_node = std::make_unique<noarr::pipelines::LambdaComputeNode>("validate");

	auto& link = validator_node->link(input_hub.to_consume(noarr::pipelines::Device::HOST_INDEX));

	validator_node->advance([&, &validator_node = *validator_node, structure]() {
		link.envelope->structure = structure;
		auto bag = noarr::pipelines::bag_from_link(link);

		validator.validate(bag);

		validator_node.callback();
	});

	return std::move(validator_node);
}

template <typename validator_t, typename structure_t>
auto construct_validator(validator_t& validator, const structure_t& structure, size_t buffer_size)
{
	auto validator_hub = construct_hub<structure_t>(buffer_size);
	auto validator_node = construct_validator(validator, structure, *validator_hub);

	return std::make_pair(std::move(validator_hub), std::move(validator_node));
}

class smart_ptr_scheduler : public noarr::pipelines::SimpleScheduler
{
	using node_ptr = std::unique_ptr<noarr::pipelines::Node>;
	std::vector<std::unique_ptr<noarr::pipelines::Node>> owning_nodes_;

public:
	void add_smart(node_ptr node)
	{
		noarr::pipelines::SimpleScheduler::add(*node);
		owning_nodes_.emplace_back(std::move(node));
	}

	smart_ptr_scheduler& operator<<(node_ptr node)
	{
		add_smart(std::move(node));

		return *this;
	}
};

class timer_cuda_compute_node : public noarr::pipelines::LambdaCudaComputeNode
{
public:
	duration_ptr advance_duration;

	timer_cuda_compute_node(const std::string& label,
							noarr::pipelines::Device::index_t device_index = noarr::pipelines::Device::DEVICE_INDEX)
		: LambdaCudaComputeNode(label, device_index)
	{
		advance_duration = std::make_shared<std::chrono::duration<double>>();
	}

protected:
	virtual void __internal__advance_cuda() override
	{
		auto start = std::chrono::steady_clock::now();
		noarr::pipelines::LambdaCudaComputeNode::__internal__advance_cuda();
		*advance_duration = std::chrono::steady_clock::now() - start;
	}
};
