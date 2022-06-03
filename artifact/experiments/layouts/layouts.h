#pragma once

#include "row_col_structure.h"
#include "row_row_structure.h"

template <char I, char J, char X, char Y, typename T>
auto resize(noarr::sized_vector<X, noarr::sized_vector<Y, noarr::scalar<T>>> structure, size_t size_I, size_t size_J)
{
	return structure | noarr::set_length<I>(size_I) | noarr::set_length<J>(size_J);
}
