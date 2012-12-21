#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>

#include "OpenCL.hpp"
#include "Backtrace.hpp"

/* OpenCL-OpenGL interoperability */
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/CGLDevice.h>
#include <OpenCL/cl_gl_ext.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <GL/glx.h>
#include <CL/cl_gl.h>
#endif

#define TYPE CL_DEVICE_TYPE_GPU

#define ERROR(msg) do{\
	cerr << "Error at " << __FILE__ << ':' << __LINE__ \
	     << "\n  " << msg << endl; \
	xromm::bt(); \
	exit(1); \
	}while(0)

#define CHECK_CL \
	if (err_ != CL_SUCCESS) {\
		cerr << "OpenCL error at " << __FILE__ << ':' << __LINE__ \
	         << "\n  " << err_ << ' ' << opencl_error(err_) << endl; \
		xromm::bt(); \
		exit(1); \
	}

using namespace std;

static bool inited_ = false;
static bool gl_inited_ = false;

#if defined(__APPLE__) || defined(__MACOSX)
static CGLShareGroupObj share_group_;
#elif defined(_WIN32)
// TODO: implement this
#else
static GLXContext glx_context_;
static Display* glx_display_;
#endif

static cl_int err_;
static cl_context context_;
static cl_device_id devices_[1];
static cl_command_queue queue_;

static const char* opencl_error(cl_int err)
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

static void print_platform(cl_platform_id platform)
{
	cerr << "# OpenCL Platform" << endl;

	char buffer[1024];

	err_ = clGetPlatformInfo(
				platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Version    : " << buffer << endl;

	err_ = clGetPlatformInfo(
				platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Name       : " << buffer << endl;

	err_ = clGetPlatformInfo(
				platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Vendor     : " << buffer << endl;
}

static void print_device(cl_device_id device)
{
	char buffer[1024];
	cl_bool b;
	cl_device_type t;
	cl_ulong ul;
	cl_uint ui;
	size_t s[3];

	cerr << "# OpenCL Device" << "\n";

	err_ = clGetDeviceInfo(
				device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Name          : " << buffer << "\n";

	err_ = clGetDeviceInfo(
				device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
	cerr << "# Type          : ";
	switch (t) {
		case CL_DEVICE_TYPE_CPU: cerr << "CPU\n"; break;
		case CL_DEVICE_TYPE_GPU: cerr << "GPU\n"; break;
		case CL_DEVICE_TYPE_ACCELERATOR: cerr << "Accelerator\n"; break;
		case CL_DEVICE_TYPE_DEFAULT: cerr << "Default\n"; break;
		default: cerr << "Unknown\n";
	}

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL);
	CHECK_CL
	cerr << "# Compute Cores : " << ui << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL);
	CHECK_CL
	cerr << "# Core Freq.    : " << ui << " Mhz\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Vendor        : " << buffer << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL);
	CHECK_CL
	cerr << "# Vendor ID     : " << ui << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Version       : " << buffer << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Driver Ver.   : " << buffer << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_AVAILABLE, sizeof(b), &b, NULL);
	CHECK_CL
	cerr << "# Available     : ";
	switch (b) {
		case CL_TRUE: cerr << "Yes\n"; break;
		case CL_FALSE: cerr << "No\n"; break;
		default: cerr << "Unknown\n";
	}

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(s), s, NULL);
	CHECK_CL
	cerr << "# Max Items     : ("
		 << s[0] << ',' << s[1] << ',' << s[2] << ")\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), s, NULL);
	CHECK_CL
	cerr << "# Max Groups    : " << s[0] << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ul), &ul, NULL);
	CHECK_CL
	cerr << "# Max Constant  : " << ul << " kB\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(ui), &ui, NULL);
	CHECK_CL
	cerr << "# Max Constants : " << ui << "\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, NULL);
	CHECK_CL
	cerr << "# Local Mem.    : " << (ul/1024) << " kB\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ul), &ul, NULL);
	CHECK_CL
	cerr << "# Global Mem.   : " << (ul/1024) << " kB\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ul), &ul, NULL);
	CHECK_CL
	cerr << "# Global Cache  : " << ul << " B\n";

	err_ = clGetDeviceInfo(
		device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
	CHECK_CL
	cerr << "# Extensions    : " << buffer << "\n";
}

namespace xromm { namespace opencl {

void opencl_global_gl_context()
{
#if defined(__APPLE__) || defined(__MACOSX)
	CGLContextObj glContext = CGLGetCurrentContext();
	share_group_ = CGLGetShareGroup(glContext);
	if (!share_group_) ERROR("invalid CGL sharegroup");
#elif defined(_WIN32)
// TODO: implement this
#else
	glx_context_ = glXGetCurrentContext();
	if (!glx_context_) ERROR("invalid GLX context");
	glx_display_ = glXGetCurrentDisplay();
	if (!glx_display_) ERROR("invalid GLX display");
#endif
	gl_inited_ = true;
}

cl_int opencl_global_context()
{
	if (!inited_)
	{
		if (!gl_inited_) return CL_INVALID_CONTEXT;

		/* find platform */

		cl_uint num_platforms;
		cl_platform_id platforms[1];
		err_ = clGetPlatformIDs(1, platforms, &num_platforms);
		CHECK_CL

		if (num_platforms < 1) ERROR("no OpenCL platforms found");

		print_platform(platforms[0]);

		/* find GPU device */

		cl_uint num_devices;
		err_ = clGetDeviceIDs(platforms[0], TYPE, 1, devices_, &num_devices);
		CHECK_CL

		if (num_devices < 1) ERROR("no OpenCL GPU device found");

		print_device(devices_[0]);

		/* create context */

#if defined(__APPLE__) || defined(__MACOSX)
#pragma OPENCL EXTENSION cl_APPLE_gl_sharing : enable
		cl_context_properties prop[] = { 
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			(intptr_t)share_group_,
			0 };

		/* omit the device, according to this:
		   http://www.khronos.org/message_boards/viewtopic.php?f=28&t=2548 */
		context_ = clCreateContext(prop, 0, NULL, 0, 0, &err_);
		CHECK_CL

		size_t num_gl_devices;
		err_ = clGetGLContextInfoAPPLE(
					context_, glContext,
					CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE,
					1, devices_, &num_gl_devices);

		if (num_gl_devices < 1) ERROR("no OpenCL GPU device found");
#elif defined(_WIN32)
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
		/* TODO: test this */
		cl_context_properties prop[] = { 
			CL_GL_CONTEXT_KHR,
			(cl_context_properties) wglGetCurrentContext(),
			CL_WGL_HDC_KHR,
			(cl_context_properties) wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, 
			(cl_context_properties)(platforms[0]),
			0 };

		context_ = clCreateContext(prop, 1, devices_, NULL, NULL, &err_);
		CHECK_CL
#else
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
		cl_context_properties prop[] = { 
			CL_GL_CONTEXT_KHR,
			(cl_context_properties)glXGetCurrentContext(),
			CL_GLX_DISPLAY_KHR,
			(cl_context_properties)glXGetCurrentDisplay(),
			CL_CONTEXT_PLATFORM, 
			(cl_context_properties)(platforms[0]),
			0 };

		context_ = clCreateContext(prop, 1, devices_, NULL, NULL, &err_);
		CHECK_CL
#endif

		/* create command queue */

		queue_ = clCreateCommandQueue(context_, devices_[0], 0, &err_);
		CHECK_CL

		inited_ = true;
	}
	return CL_SUCCESS;
}

Kernel::Kernel(cl_program program, const char* func)
{
	err_ = opencl_global_context();
	CHECK_CL
	reset();
	kernel_ = clCreateKernel(program, func, &err_);
	CHECK_CL
}

void Kernel::reset()
{
	arg_index_ = 0;
	grid_dim_ = 0;
	block_dim_ = 0;
}

size_t Kernel::getLocalMemSize()
{
	err_ = opencl_global_context();
	CHECK_CL
	size_t s;
	err_ = clGetDeviceInfo(devices_[0],
					CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_t), &s, NULL);
	CHECK_CL
	return s;
}

size_t* Kernel::getMaxItems()
{
	err_ = opencl_global_context();
	CHECK_CL
	size_t* s = new size_t[3];
	err_ = clGetDeviceInfo(devices_[0],
					CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(s), s, NULL);
	CHECK_CL
	return s;
}

size_t Kernel::getMaxGroups()
{
	err_ = opencl_global_context();
	CHECK_CL
	size_t s;
	err_ = clGetDeviceInfo(devices_[0],
					CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &s, NULL);
	CHECK_CL
	return s;
}

void Kernel::grid1d(size_t X)
{
	if (grid_dim_ && (grid_dim_ != 1)) {
		ERROR("Grid dimension was already set and is not 1");
	} else if (!block_dim_) {
		ERROR("Must set block dimension before grid");
	} else {
		grid_dim_ = 1;
	}
	grid_[0] = X * block_[0];
}

void Kernel::grid2d(size_t X, size_t Y)
{
	if (grid_dim_ && (grid_dim_ != 2)) {
		ERROR("Grid dimension was already set and is not 2");
	} else if (!block_dim_) {
		ERROR("Must set block dimension before grid");
	} else {
		grid_dim_ = 2;
	}
	grid_[0] = X * block_[0];
	grid_[1] = Y * block_[1];
}

void Kernel::block1d(size_t X)
{
	if (block_dim_ && (block_dim_ != 1)) {
		ERROR("Block dimension was already set and is not 1");
	} else {
		block_dim_ = 1;
	}
	block_[0] = X;
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

void Kernel::addBufferArg(const Buffer* buf)
{
	err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &buf->buffer_);
	CHECK_CL
}

void Kernel::addGLBufferArg(const GLBuffer* buf)
{
	gl_buffers.push_back(buf);
	err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &buf->buffer_);
	CHECK_CL
}

void Kernel::addImageArg(const Image* img)
{
	err_ = clSetKernelArg(kernel_, arg_index_++, sizeof(cl_mem), &img->image_);
	CHECK_CL
}

/* Dynamically allocated local memory can be added to the kernel by passing
   a NULL argument with the size of the buffer, e.g.
   http://stackoverflow.com/questions/8888718/how-to-declare-local-memory-in-opencl
*/
void Kernel::addLocalMem(size_t size)
{
	err_ = clSetKernelArg(kernel_, arg_index_++, size, NULL);
}

void Kernel::launch()
{
	//cout << grid_[0] << "," << grid_[1] << endl;
	//cout << block_[0] << "," << block_[1] << endl;
	if (!block_dim_) {
		ERROR("Block dimension is unset");
	} else if (!grid_dim_) {
		ERROR("Grid dimension is unset");
	} else if (block_dim_ != grid_dim_) {
		ERROR("Block dimension doesn't match grid dimension");
	}

	unsigned n_gl_buffers = gl_buffers.size();
	cl_mem* gl_mem = NULL;
	if (n_gl_buffers)
	{
		gl_mem = new cl_mem[n_gl_buffers];

		for (unsigned i=0; i<n_gl_buffers; i++) {
			gl_mem[i] = gl_buffers[i]->buffer_;
		}

		err_ = clEnqueueAcquireGLObjects(
				queue_, n_gl_buffers, gl_mem, 0, NULL, NULL);
		CHECK_CL
	}

	err_ = clEnqueueNDRangeKernel(
			queue_, kernel_, grid_dim_, NULL,
			grid_, block_, 0, NULL, NULL);
	CHECK_CL

	if (n_gl_buffers)
	{
		err_ = clEnqueueReleaseGLObjects(
				queue_, n_gl_buffers, gl_mem, 0, NULL, NULL);
		CHECK_CL

		delete gl_mem;
	}
}

void Kernel::setArg(cl_uint i, size_t size, const void* value)
{
	err_ = clSetKernelArg(kernel_, i, size, value);
	CHECK_CL
}

Program::Program() { compiled_ = false; }

Kernel* Program::compile(const char* code, const char* func)
{
	if (!compiled_)
	{
		err_ = opencl_global_context();
		CHECK_CL

		size_t len = strlen(code);
		program_ = clCreateProgramWithSource(context_, 1, &code, &len, &err_);
		CHECK_CL

		err_ = clBuildProgram(program_, 1, devices_, NULL, NULL, NULL);
		if (err_ == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			err_ = clGetProgramBuildInfo(
					program_, devices_[0], CL_PROGRAM_BUILD_LOG,
					0, NULL, &log_size);
			CHECK_CL
			char* build_log = (char*)malloc(log_size+1);
			if (!build_log) ERROR("malloc for build log");
			err_ = clGetProgramBuildInfo(
					program_, devices_[0], CL_PROGRAM_BUILD_LOG,
					log_size, build_log, NULL);
			CHECK_CL
			build_log[log_size] = '\0';
			cerr << "OpenCL build failure for kernel function '" << func
			     << "':\n" << build_log << endl;
			free(build_log);
			exit(1);
		} else {
			CHECK_CL
		}

		compiled_ = true;
	}

	return new Kernel(program_, func);
}

Buffer::Buffer(size_t size, cl_mem_flags access)
{
	err_ = opencl_global_context();
	CHECK_CL
	size_ = size;
	access_ = access;
	buffer_ = clCreateBuffer(context_, access, size, NULL, &err_);
	CHECK_CL
}

Buffer::~Buffer()
{
	err_ = clReleaseMemObject(buffer_);
	CHECK_CL
}

void Buffer::read(const void* buf, size_t size) const
{
	if (size == 0) size = size_;
	err_ = clEnqueueWriteBuffer(
			queue_, buffer_, CL_TRUE, 0, size, buf, 0, NULL, NULL);
	CHECK_CL
}

void Buffer::write(void* buf, size_t size) const
{
	if (size == 0) size = size_;
	err_ = clEnqueueReadBuffer(
			queue_, buffer_, CL_TRUE, 0, size, buf, 0, NULL, NULL);
	CHECK_CL
}

void Buffer::copy(const Buffer* dst, size_t size) const
{
	if (size == 0) size = size_;
	if (size > dst->size_)
		ERROR("Destination buffer does not have enough room!");
	err_ = clEnqueueCopyBuffer(
			queue_, dst->buffer_, buffer_, 0, 0, size, 0, NULL, NULL);
	CHECK_CL
}

void Buffer::zero() const
{
#ifdef CL_VERSION_1_2
	char c = 0x00;
	err_ = clEnqueueFillBuffer(queue_, buffer_, &c, 1, 0, size_, 0, NULL, NULL);
	CHECK_CL
#else
	char* tmp = (char*)new char[size_];
	memset(tmp, 0x00, size_);
	err_ = clEnqueueWriteBuffer(
			queue_, buffer_, CL_TRUE, 0, size_, (void*)tmp, 0, NULL, NULL);
	CHECK_CL
	delete tmp;
#endif
}

GLBuffer::GLBuffer(GLuint pbo, cl_mem_flags access)
{
	err_ = opencl_global_context();
	CHECK_CL
	access_ = access;
	buffer_ = clCreateFromGLBuffer(context_, access, pbo, &err_);
	CHECK_CL
}

GLBuffer::~GLBuffer()
{
	err_ = clReleaseMemObject(buffer_);
	CHECK_CL
}

Image::Image(size_t* dims, cl_image_format *format, cl_mem_flags access)
{
	err_ = opencl_global_context();
	CHECK_CL
	dims_[0] = dims[0];
	dims_[1] = dims[1];
	dims_[2] = dims[2];
	access_ = access;

	if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
		ERROR("Image object must have non-zero dimensions");

#ifdef CL_VERSION_1_2
	cl_image_desc desc;

	if (dims[2] == 1) {
		desc.image_type  = CL_MEM_OBJECT_IMAGE2D;
	} else {
		desc.image_type  = CL_MEM_OBJECT_IMAGE3D;
	}

	desc.image_width       = dims[0];
	desc.image_height      = dims[1];
	desc.image_depth       = dims[2];
	desc.image_row_pitch   = 0;
	desc.image_slice_pitch = 0;
//	desc.num_mip_level     = 0;
	desc.num_samples       = 0;
	desc.buffer            = NULL;

	image_ = clCreateImage(context_, access, format, &desc, NULL, &err_);
#else
	if (dims[2] == 1) {
		image_ = clCreateImage2D(context_,
					access, format, dims[0], dims[1], 0, NULL, &err_);	
	} else {
		image_ = clCreateImage3D(
					context_, access, format,
					dims[0], dims[1], dims[2],
					0, 0, NULL, &err_);	
	}
#endif

	CHECK_CL
}

Image::~Image()
{
	err_ = clReleaseMemObject(image_);
	CHECK_CL
}

void Image::read(const void* buf) const
{
	size_t origin[3] = {0,0,0};
	err_ = clEnqueueWriteImage(
			queue_, image_, CL_TRUE, origin, dims_, 0, 0, buf, 0, NULL, NULL);
	CHECK_CL
}

void Image::write(void* buf) const
{
	size_t origin[3] = {0,0,0};
	err_ = clEnqueueReadImage(
			queue_, image_, CL_TRUE, origin, dims_, 0, 0, buf, 0, NULL, NULL);
	CHECK_CL
}

} } // namespace xromm::opencl
