#include "examples.h"

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);
	run_stencil(args);
	return 0;
}