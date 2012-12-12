#ifndef XROMM_OPENCL_HPP
#define XROMM_OPENCL_HPP

#include <iostream>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>
#else
#include <GL/gl.h>
#include <CL/opencl.h>
#endif

/* OpenCL-OpenGL interoperability */
#pragma OPENCL EXTENSION cl_khr_gl_sharing enable

namespace xromm { namespace opencl {

void init();

class Kernel
{
public:
	Kernel(cl_program program, const char* func);
	void reset();

	static size_t getLocalMemSize();
	static size_t* getMaxItems();
	static size_t getMaxGroups();

	void grid1d(size_t X);
	void block1d(size_t X);
	void grid2d(size_t X, size_t Y);
	void block2d(size_t X, size_t Y);
	void launch();

	void addBufferArg(const ReadBuffer* buf);
	void addBufferArg(const WriteBuffer* buf);
	void addLocalMem(size_t size);

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

class Buffer
{
public:
	Buffer(size_t size);
	~Buffer();
	void read(const void* buf, size_t size=0) const;
	void write(void* buf, size_t size=0) const;
	void copy(const Buffer* buf, size_t size=0) const;
	friend class Kernel;
protected:
	size_t size_;
	cl_mem buffer_;
};

class ReadBuffer : public Buffer
{
public:
	ReadBuffer(size_t size);
};

class WriteBuffer : public Buffer
{
public:
	WriteBuffer(size_t size);
};

} } // namespace xromm::opencl

#endif // XROMM_OPENCL_HPP
