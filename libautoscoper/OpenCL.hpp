#ifndef XROMM_OPENCL_HPP
#define XROMM_OPENCL_HPP

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

namespace xromm { namespace opencl {

cl::Buffer* device_alloc(size_t size, cl_mem_flags flags);
void copy_to_device(cl::Buffer* dst, const void* src, size_t size);
void copy_from_device(void* dst, const cl::Buffer* src, size_t size);

class Kernel;

class Program
{
public:
	Program();
    Kernel* compile(const char* code, const char* kernel);
	Kernel* compileFile(const char* filename, const char* kernel);
protected:
	cl::Program program_;
	bool compiled_;
};

class Kernel
{
public:
	Kernel(cl::Program& program, const char* name);
	void grid2d(size_t X, size_t Y);
	void block2d(size_t X, size_t Y);
	void bind(void* value, size_t size);
	void launch();
protected:
	cl::Kernel kernel_;
	cl_uint arg_index_;
	cl::NDRange grid_;
	cl_uint grid_dim_;
	cl::NDRange block_;
	cl_uint block_dim_;
};

} } // namespace xromm::opencl

#endif // XROMM_OPENCL_HPP
