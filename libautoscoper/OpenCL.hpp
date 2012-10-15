
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

namespace xromm { namespace opencl {

void init();

class Program
{
public:
	Program() {};
	void compile(const char* filename, const char* kernel);
protected:
	cl::Program program_;
	bool compiled_ = false;
}

class Kernel
{
public:
	Kernel(const char* name);
	grid2d(size_t X, size_t Y);
	block2d(size_t X, size_t Y);
	bind(const void* value, size_t size);
	launch();
protected:
	cl::Kernel kernel_;
	cl_uint arg_index_ = 0;
	cl::NDRange grid_;
	cl_uint grid_dim_ = 0;
	cl::NDRange block_;
	cl_uint blockd_dim_ = 0;
}

} } // namespace xromm::opencl
