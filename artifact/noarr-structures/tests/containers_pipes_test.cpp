#include <catch2/catch.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include "noarr/structures_extended.hpp"

TEST_CASE("Pipes vector", "[resizing]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10);

	SECTION("size check 1") {
		REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
	}

	auto v3 = v  | noarr::set_length<'x'>(20);
	auto v4 = v2 | noarr::set_length<'x'>(30);

	SECTION("size check 2") {
		REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
		REQUIRE((v3 | noarr::get_length<'x'>()) == 20);
		REQUIRE((v4 | noarr::get_length<'x'>()) == 30);
	}

	SECTION("check is_cube 2") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE( noarr::is_cube<decltype(v2)>::value);
		REQUIRE( noarr::is_cube<decltype(v3)>::value);
		REQUIRE( noarr::is_cube<decltype(v4)>::value);
	}

	auto v5 = v4 | noarr::set_length<'x'>(10);

	SECTION("size check 3") {
		REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
		REQUIRE((v3 | noarr::get_length<'x'>()) == 20);
		REQUIRE((v4 | noarr::get_length<'x'>()) == 30);
		REQUIRE((v5 | noarr::get_length<'x'>()) == 10);
	}

	SECTION("check is_cube 3") {
		REQUIRE(noarr::is_cube<decltype(v5)>::value);
	}
}

TEST_CASE("Pipes vector2", "[is_trivial]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10);

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::set_length<'x'>(20);
	auto v4 = v2 | noarr::set_length<'x'>(30);

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}

TEST_CASE("Pipes array", "[is_trivial]")
{
	noarr::array<'x', 1920, noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10);

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::set_length<'x'>(20);
	auto v4 = v2 | noarr::set_length<'x'>(30);

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}


TEST_CASE("Pipes do not affect bitwise or", "[is_trivial]")
{
	auto num = 3 | 12;

	REQUIRE(num == 15);
}