#pragma once

#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/structs.hpp"

template <char Dim, std::size_t chunk>
struct rr_block_indexer_i : private noarr::contain<std::size_t, std::size_t>
{
	using base = noarr::contain<std::size_t, std::size_t>;

	static constexpr std::tuple<> sub_structures() { return {}; }

	using description = noarr::struct_description<
		noarr::char_pack<'r', 'r', '_', 'b', 'l', 'o', 'c', 'k', '_', 'i', 'n', 'd', 'e', 'x', 'e', 'r', '_', 'i'>,
		noarr::dims_impl<Dim>, noarr::dims_impl<>>;

	constexpr rr_block_indexer_i() = default;
	constexpr rr_block_indexer_i(std::size_t i_length, std::size_t j_length) : base(i_length, j_length) {};

	constexpr auto construct() const
	{
		return rr_block_indexer_i<Dim, chunk>(base::template get<0>(), base::template get<1>());
	}

	static constexpr std::size_t size() { return 0; }
	constexpr std::size_t length() const { return base::template get<0>(); }
	constexpr std::size_t offset(std::size_t i) const
	{
		return (i / chunk) * chunk * chunk * (base::template get<1>() / chunk) + (i % chunk) * chunk;
	}
};

template <char Dim, std::size_t chunk>
struct rr_block_indexer_j : private noarr::contain<std::size_t>
{
	using base = noarr::contain<std::size_t>;

	static constexpr std::tuple<> sub_structures() { return {}; }

	using description = noarr::struct_description<
		noarr::char_pack<'r', 'r', '_', 'b', 'l', 'o', 'c', 'k', '_', 'i', 'n', 'd', 'e', 'x', 'e', 'r', '_', 'j'>,
		noarr::dims_impl<Dim>, noarr::dims_impl<>>;

	constexpr rr_block_indexer_j() = default;
	constexpr rr_block_indexer_j(std::size_t j_length) : base(j_length) {};

	constexpr auto construct() const { return rr_block_indexer_j<Dim, chunk>(base::template get<0>()); }

	static constexpr std::size_t size() { return 0; }
	constexpr std::size_t length() const { return base::template get<0>(); }
	static constexpr std::size_t offset(std::size_t j) { return (j / chunk) * chunk * chunk + (j % chunk); }
};

template <typename T, typename... KS>
struct row_row_curve_get_t;

template <typename T>
struct row_row_curve_get_t<T>
{
	using type = typename T::template get_t<>;
};

template <typename T>
struct row_row_curve_get_t<T, void>
{
	using type = typename T::template get_t<void>;
};

template <typename T, typename TH1, typename TH2>
struct row_row_curve : private noarr::contain<T, TH1, TH2>
{
	using base = noarr::contain<T, TH1, TH2>;

	constexpr auto sub_structures() const
	{
		return std::tuple_cat(base::template get<0>().sub_structures(),
							  std::make_tuple(base::template get<1>(), base::template get<2>()));
	}

	using description =
		noarr::struct_description<noarr::char_pack<'r', 'o', 'w', '_', 'r', 'o', 'w', '_', 'c', 'u', 'r', 'v', 'e'>,
								  noarr::dims_impl<>, noarr::dims_impl<>, noarr::type_param<T>, noarr::type_param<TH1>,
								  noarr::type_param<TH2>>;

	template <typename... KS>
	using get_t = typename row_row_curve_get_t<T, KS...>::type;

	constexpr row_row_curve() = default;
	explicit constexpr row_row_curve(T sub_structure, TH1 sub_structure1, TH2 sub_structure2)
		: base(sub_structure, sub_structure1, sub_structure2)
	{}

	template <typename T2, typename TH3, typename TH4>
	constexpr auto construct(T2 sub_structure, TH3 sub_structure1, TH4 sub_structure2) const
	{
		return row_row_curve<decltype(this->base::template get<0>().construct(sub_structure)), TH3, TH4>(
			base::template get<0>().construct(sub_structure), sub_structure1, sub_structure2);
	}

	constexpr std::size_t size() const { return base::template get<0>().size(); }
	constexpr std::size_t offset() const
	{
		return base::template get<0>().offset(base::template get<1>().offset() + base::template get<2>().offset());
	}
};

template <char Dim1, char Dim2, typename T, std::size_t chunk>
using rr_curve = row_row_curve<T, rr_block_indexer_i<Dim1, chunk>, rr_block_indexer_j<Dim2, chunk>>;

template <char I, char J, char X, char Y, typename T, std::size_t chunk>
auto resize(rr_curve<X, Y, noarr::sized_vector<'a', noarr::scalar<T>>, chunk> structure, size_t size_I, size_t size_J)
{
	// in future to support multiple dims, add map<char,size_t> to pass dim sizes
	if (I != X)
		std::swap(size_I, size_J);

	return rr_curve<X, Y, noarr::sized_vector<'a', noarr::scalar<T>>, chunk>(
		noarr::sized_vector<'a', noarr::scalar<T>>(noarr::scalar<T>(), size_I * size_J),
		rr_block_indexer_i<X, chunk>(size_I, size_J), rr_block_indexer_j<Y, chunk>(size_J));
}
