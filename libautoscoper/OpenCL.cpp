#include <iostream>
#include <fstream>
#include "OpenCL.hpp"

#define TYPE CL_DEVICE_TYPE_GPU

#define CL_ERROR(e) \
	std::cerr << "OpenCL error at " << __FILE__ << ':' << __LINE__ \
	          << "\n  " << e.err() << ' ' << opencl_error(e.err()) << std::endl; \
	exit(1);

namespace xromm { namespace opencl {

static bool inited_ = false;
static cl::Context context_;
static cl::vector<cl::Device> devices_;
static cl::Device device_;
static cl::CommandQueue queue_;

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

static void init_context()
{
	cl::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.size() == 0)
		throw cl::Error(1, "No OpenCL platforms were found");

	int platformID = -1;

	for(unsigned int i = 0; i < platforms.size(); i++) {
		try {
			platforms[i].getDevices(TYPE, &devices_);
			platformID = i;
			break;
		} catch(cl::Error e) {
			CL_ERROR(e);
		}
	}

	if (platformID == -1)
		throw cl::Error(1, "No compatible OpenCL platform found");

	cl_context_properties cps[3] = { 
		CL_CONTEXT_PLATFORM, 
		(cl_context_properties)(platforms[platformID])(), 
		0 };

	context_ = cl::Context(TYPE, cps);
}

static void init()
{
	if (!inited_) {
		try {
			init_context();
			cl::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();
			device_ = devices[0];
			queue_ = cl::CommandQueue(context_, device_);
		} catch (cl::Error e) {
			CL_ERROR(e);
		}
		std::cout << "OpenCL: init" << std::endl;
		inited_ = true;
	}
}

cl::Buffer* device_alloc(size_t size, cl_mem_flags flags)
{
	init();
	try {
		return new cl::Buffer(context_, flags, size);
	} catch (cl::Error e) {
		CL_ERROR(e);
	}
}

void copy_to_device(cl::Buffer* dst, const void* src, size_t size)
{
	init();
	try {
		queue_.enqueueWriteBuffer(*dst, CL_TRUE, 0, size, src);
	} catch (cl::Error e) {
		CL_ERROR(e);
	}
}

void copy_from_device(void* dst, const cl::Buffer* src, size_t size)
{
	init();
	try {
		queue_.enqueueReadBuffer(*src, CL_TRUE, 0, size, dst);
	} catch (cl::Error e) {
		CL_ERROR(e);
	}
}

Program::Program() { init(); compiled_ = false; }

Kernel* Program::compile(const char* filename, const char* kernel)
{
	if (!compiled_) {
		std::ifstream srcFile(filename);
		std::string code(
			(std::istreambuf_iterator<char>(srcFile)),
			std::istreambuf_iterator<char>());
		cl::Program::Sources src(1, std::make_pair(code.c_str(), code.length()+1));
		program_ = cl::Program(context_, src);

		// Build program for these specific devices
		try {
			program_.build(devices_);
		} catch(cl::Error e) {
			if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::cerr << "Build log:\n" << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
			}
			CL_ERROR(e);
		} 

		compiled_ = true;
	}

	return new Kernel(program_, kernel);
}

Kernel::Kernel(cl::Program& program, const char* name)
{
	init();
	arg_index_ = 0;
	grid_dim_ = 0;
	block_dim_ = 0;
	kernel_ = cl::Kernel(program, name);
}

void Kernel::grid2d(size_t X, size_t Y)
{
	if (grid_dim_ && (grid_dim_ != 2))
		throw cl::Error(1, "Grid dimension was already set and is not 2");
	else grid_dim_ = 2;
	grid_ = cl::NDRange(X, Y);
}

void Kernel::block2d(size_t X, size_t Y)
{
	if (block_dim_ && (block_dim_ != 2))
		throw cl::Error(1, "Block dimension was already set and is not 2");
	else block_dim_ = 2;
	block_ = cl::NDRange(X, Y);
}

void Kernel::bind(void* value, size_t size)
{
	kernel_.setArg(arg_index_++, size, value);
}

void Kernel::launch()
{
	if (!block_dim_)
		throw cl::Error(1, "Block dimension is unset");
	else if (!grid_dim_)
		throw cl::Error(1, "Grid dimension is unset");
	else if (block_dim_ != grid_dim_)
		throw cl::Error(1, "Block dimension doesn't match grid dimension");
	queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, grid_, block_);
}

} } // namespace xromm::opencl
