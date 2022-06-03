#pragma once
#include <memory>

#include <noarr/pipelines.hpp>
#include <noarr/structures-pipelines.hpp>

#include "transformer.h"

template <dim_t X, dim_t Y, typename transform_struct_t, typename in_link_t, typename out_link_t>
auto construct_host_transformer_2d_node(in_link_t& consume_link, out_link_t& produce_link)
{
	auto transformer = std::make_unique<noarr::pipelines::LambdaComputeNode>("transformer");
	auto& input_link = transformer->link(consume_link);
	auto& output_link = transformer->link(produce_link);

	transformer->advance([&, &transformer = *transformer]() {
		auto input_bag = noarr::pipelines::bag_from_link(input_link);
		auto x_length = input_bag.structure().template get_length<X>();
		auto y_length = input_bag.structure().template get_length<Y>();

		output_link.envelope->structure = resize<X, Y>(transform_struct_t(), x_length, y_length);

		auto output_bag = noarr::pipelines::bag_from_link(output_link);

		base_host_transformer_2d::transform<X, Y>(input_bag, output_bag);

		transformer.callback();
	});

	return transformer;
}

template <dim_t X, dim_t Y, typename out_structure_t, typename in_hub_t, typename scheduler_t>
auto get_host_transformer_2d_hub_and_node(in_hub_t& input_hub, size_t output_hub_size)
{
	auto output_hub = noarr::pipelines::Hub<out_structure_t>(output_hub_size);
	input_hub.allocate_envelopes(noarr::pipelines::Device::HOST_INDEX, 2);

	auto transformer = construct_host_transformer_2d_node<X, Y, out_structure_t>(
		input_hub.to_consume(noarr::pipelines::Device::HOST_INDEX),
		output_hub.to_produce(noarr::pipelines::Device::HOST_INDEX));

	return std::make_pair(transformer, output_hub);
}
