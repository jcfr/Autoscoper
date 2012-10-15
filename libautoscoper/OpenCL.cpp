#include "OpenCL.hpp"

#define TYPE CL_DEVICE_TYPE_GPU

namespace xromm { namespace opencl {

static cl::Context context_;
static cl::Device device_;
static cl::CommandQueue queue_;

static void init_context()
{
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.size() == 0)
        throw cl::Error(1, "No OpenCL platforms were found");

    int platformID = -1;

    for(unsigned int i = 0; i < platforms.size(); i++) {
        try {
            cl::vector<cl::Device> devices;
            platforms[i].getDevices(TYPE, &devices);
            platformID = i;
            break;
        } catch(cl::Error e) {
            continue; 
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

void init()
{
	init_context();
    cl::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();
	device_ = devices[0];
	queue_ = cl::CommandQueue(context_, device_);
}

void Program::compile(const char* filename, const char* kernel)
{
	if (!compiled_) {
		std::ifstream srcFile(filename);
		std::string code(
			(std::istreambuf_iterator<char>(srcFile)),
			std::istreambuf_iterator<char>());
		cl::Program::Sources src(1, std::make_pair(code.c_str(), code.length()+1));
		program_ = cl::Program(context_, src);

        // Build program for these specific devices
        try{
            program_.build(device_);
        } catch(cl::Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:" << std::endl << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 

		compiled_ = true;
	}

	return Kernel(program_, kernel);
}

Kernel::Kernel(Program& program, const char* name)
{
	kernel_ = cl::Kernel(program_, name);
}

void Kernel::grid2d(size_t X, size_t Y)
{
	if (grid_dim_ && (grid_dim_ != 2)) error;
	else grid_dim_ = 2;
	grid_ = cl::NDRange(X, Y);
}

void Kernel::block2d(size_t X, size_t Y)
{
	if (block_dim_ && (block_dim_ != 2)) error;
	else block_dim_ = 2;
	block_ = cl::NDRange(X, Y);
}

void Kernel::bind(const void* value, size_t size)
{
	kernel_.set_arg(arg_index_++, size, value);
}

void Kernel::launch()
{
	if (!block_dim_) error;
	else if (!grid_dim_) error;
	else if (block_dim_ != grid_dim_) error;
	queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, grid_, block_);
}

} } // namespace xromm::opencl
