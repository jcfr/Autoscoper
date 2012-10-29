#include <iostream>
#include <fstream>
#include "OpenCL.hpp"

#define TYPE CL_DEVICE_TYPE_GPU

#define ERROR(msg) \
	std::cerr << "Error at " << __FILE__ << ':' << __LINE__ \
	<< "\n  " << msg << std::endl; \
	exit(1)

#define CHECK_CL \
	if (err_ != CL_SUCCESS) {\
		std::cerr << "OpenCL error at " << __FILE__ << ':' << __LINE__ \
	          << "\n  " << err_ << ' ' << opencl_error(err_) << std::endl; \
		exit(1);\
	}

namespace xromm { namespace opencl {

static bool inited_ = false;
static cl_int err_;
static cl_context context_;
static cl_device_id devices_[1];
static cl_command_queue queue_;

static const char* opencl_error(cl_uint err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

static void init()
{
	if (!inited_)
	{
		/* find platform */

		cl_uint num_platforms;
		cl_platform_id platforms[1];
		err_ = clGetPlatformIDs(1, platforms, &num_platforms);
		CHECK_CL

		if (num_platforms != 1) ERROR("no OpenCL platforms found");

		/* find GPU device */

		cl_uint num_devices;
		err_ = clGetDeviceIDs(platforms[0], TYPE, 1, devices_, &num_devices);
		CHECK_CL

		if (num_devices != 1) ERROR("no OpenCL GPU device found");

		/* create context */

		cl_context_properties prop[3] = { 
			CL_CONTEXT_PLATFORM, 
			(cl_context_properties)(platforms[0]),
			0 };

		context_ = clCreateContext(prop, 1, devices_, NULL, NULL, &err_);
		CHECK_CL

		/* create command queue */

		queue_ = clCreateCommandQueue(context_, devices_[0], 0, &err_);
		CHECK_CL

		std::cout << "OpenCL: init" << std::endl;
		inited_ = true;
	}
}

Kernel::Kernel(cl_program program, const char* func)
{
	init();
	arg_index_ = 0;
	grid_dim_ = 0;
	block_dim_ = 0;
	kernel_ = clCreateKernel(program, func, &err_);
	CHECK_CL
}

void Kernel::grid2d(size_t X, size_t Y)
{
	if (grid_dim_ && (grid_dim_ != 2)) {
		ERROR("Grid dimension was already set and is not 2");
	} else {
		grid_dim_ = 2;
	}
	grid_[0] = X;
	grid_[1] = Y;
}

void Kernel::block2d(size_t X, size_t Y)
{
	if (block_dim_ && (block_dim_ != 2)) {
		ERROR("Block dimension was already set and is not 2");
	} else {
		block_dim_ = 2;
	}
	block_[0] = X;
	block_[1] = Y;
}

void Kernel::addBufferArg(const ReadBuffer* buf)
{
	err_ = clSetKernelArg(kernel_, arg_index_++, buf->size_, buf->buffer_);
	CHECK_CL
}

void Kernel::addBufferArg(const WriteBuffer* buf)
{
	err_ = clSetKernelArg(kernel_, arg_index_++, buf->size_, buf->buffer_);
	CHECK_CL
}

void Kernel::launch()
{
	if (!block_dim_) {
		ERROR("Block dimension is unset");
	} else if (!grid_dim_) {
		ERROR("Grid dimension is unset");
	} else if (block_dim_ != grid_dim_) {
		ERROR("Block dimension doesn't match grid dimension");
	}
	err_ = clEnqueueNDRangeKernel(
			queue_, kernel_, grid_dim_, NULL,
			grid_, block_, 0, NULL, NULL);
	CHECK_CL
}

void Kernel::setArg(cl_uint i, size_t size, const void* value)
{
	err_ = clSetKernelArg(kernel_, i, size, value);
	CHECK_CL
}

Program::Program() { init(); compiled_ = false; }

Kernel* Program::compile(const char* code, const char* func)
{
	if (!compiled_) {
		// Build program for these specific devices
		std::cout << code << std::endl;
		size_t len = strlen(code);
		program_ = clCreateProgramWithSource(context_, 1, &code, &len, &err_);
		CHECK_CL

		err_ = clBuildProgram(program_, 1, devices_, NULL, NULL, NULL);
		if (err_ == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			cl_int err = clGetProgramBuildInfo(
					program_, devices_[0], CL_PROGRAM_BUILD_LOG,
					0, NULL, &log_size);
			char* build_log = (char*)malloc(log_size+1);
			err = clGetProgramBuildInfo(
					program_, devices_[0], CL_PROGRAM_BUILD_LOG,
					log_size, build_log, NULL);
			build_log[log_size] = '\0';
			std::cerr << "Build log:\n" << build_log << std::endl;
			free(build_log);
		}
		CHECK_CL

		compiled_ = true;
	}

	return new Kernel(program_, func);
}

ReadBuffer::ReadBuffer(size_t size)
{
	init();
	size_ = size;
	buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, size, NULL, &err_);
	CHECK_CL
}

ReadBuffer::~ReadBuffer()
{
	err_ = clReleaseMemObject(buffer_);
	CHECK_CL
}

void ReadBuffer::read(const void* buf) const
{
	err_ = clEnqueueWriteBuffer(
			queue_, buffer_, CL_TRUE, 0, size_, buf, 0, NULL, NULL);
	CHECK_CL
}

void ReadBuffer::write(const WriteBuffer* buf) const
{
	if (size_ != buf->size_) {
		ERROR("Buffers have mismatching sizes");
	}
	err_ = clEnqueueCopyBuffer(
			queue_, buf->buffer_, buffer_, 0, 0, size_, 0, NULL, NULL);
	CHECK_CL
}

WriteBuffer::WriteBuffer(size_t size)
{
	init();
	size_ = size;
	buffer_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, size, NULL, &err_);
	CHECK_CL
}

WriteBuffer::~WriteBuffer()
{
	err_ = clReleaseMemObject(buffer_);
	CHECK_CL
}

void WriteBuffer::write(void* buf) const
{
	err_ = clEnqueueReadBuffer(
			queue_, buffer_, CL_TRUE, 0, size_, buf, 0, NULL, NULL);
	CHECK_CL
}

} } // namespace xromm::opencl
