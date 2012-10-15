
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

class OpenCL
{
public:
	OpenCL(const char* filename, const char* kernel, cl_device_type type);
	grid2d(size_t X, size_t Y);
	block2d(size_t X, size_t Y);
	bind(const void* value, size_t size);
	launch();
protected:
	getPlatform(cl_device_type type);
	buildProgram(const char* filename);

	cl::Context _context;
	cl::Program _program;
	cl::CommandQueue _queue;
	cl::Kernel _kernel;

	cl_uint arg_index = 0;

	cl::NDRange grid;
	cl_uint grid_dim = 0;

	cl::NDRange block;
	cl_uint blockd_dim = 0;
}
