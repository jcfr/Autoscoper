#ifndef XROMM_OPENCL_HPP
#define XROMM_OPENCL_HPP

#include <iostream>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>
#else
#include <GL/gl.h>
#include <CL/opencl.h>
#endif // !__APPLE__

namespace xromm { namespace opencl {

class ReadBuffer;
class WriteBuffer;

class Kernel
{
public:
	Kernel(cl_program program, const char* func);
	void grid2d(size_t X, size_t Y);
	void block2d(size_t X, size_t Y);
	void launch();

	void addBufferArg(const ReadBuffer* buf);
	void addBufferArg(const WriteBuffer* buf);

	template<typename T> void addArg(T& value)
	{
		setArg(arg_index_++, sizeof(T), (const void*)(&value));
	}

protected:
	void setArg(cl_uint i, size_t size, const void* value);
	cl_kernel kernel_;
	cl_uint arg_index_;
	size_t grid_[3];
	cl_uint grid_dim_;
	size_t block_[3];
	cl_uint block_dim_;
};

class Program
{
public:
	Program();
    Kernel* compile(const char* code, const char* func);
protected:
	cl_program program_;
	bool compiled_;
};

class ReadBuffer
{
public:
	ReadBuffer(size_t size);
	~ReadBuffer();
	void read(const void* buf) const;
	void write(const WriteBuffer* buf) const;
	friend class Kernel;
protected:
	size_t size_;
	cl_mem buffer_;
};

class WriteBuffer
{
public:
	WriteBuffer(size_t size);
	~WriteBuffer();
	void write(void* buf) const;
	friend class ReadBuffer;
	friend class Kernel;
protected:
	size_t size_;
	cl_mem buffer_;
};


} } // namespace xromm::opencl

#endif // XROMM_OPENCL_HPP
