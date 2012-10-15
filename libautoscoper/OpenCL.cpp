#include "OpenCL.hpp"

void OpenCL::getPlatform(cl_device_type type)
{
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.size() == 0)
        throw cl::Error(1, "No OpenCL platforms were found");

    int platformID = -1;

    for(unsigned int i = 0; i < platforms.size(); i++) {
        try {
            cl::vector<cl::Device> devices;
            platforms[i].getDevices(type, &devices);
            platformID = i;
            break;
        } catch(cl::Error e) {
            continue; 
        }
    }

    if (platformID == -1)
        throw cl::Error(1, "No compatible OpenCL platform found");

    _platform = platforms[platformID];
}

void OpenCL::buildProgram(const char* filename)
{
	std::ifstream srcFile(filename);
	std::string code(
		(std::istreambuf_iterator<char>(srcFile)),
		std::istreambuf_iterator<char>());
	cl::Program::Sources src(1, std::make_pair(code.c_str(), code.length()+1));
	_program = cl::Program(_context, src);

     cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	_queue = CommandQueue(context, devices[0]);
    
        // Build program for these specific devices
        try{
            _program.build(devices);
        } catch(cl::Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:" << std::endl << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 

}

OpenCL::OpenCL(const char* filename, const char* kernel, cl_device_type type)
{
	getPlatform(type);
	buildProgram(filename);
	_kernel = Kernel(_program, kernel);
}

void OpenCL::grid2d(size_t X, size_t Y)
{
	if (grid_dim && (grid_dim != 2)) error;
	else grid_dim = 2;
	grid = cl::NDRange(X, Y);
}

void OpenCL::block2d(size_t X, size_t Y)
{
	if (block_dim && (block_dim != 2)) error;
	else block_dim = 2;
	block = cl::NDRange(X, Y);
}

void OpenCL::bind(const void* value, size_t size)
{
	kernel.set_arg(arg_index++, size, value);
}

void OpenCL::launch()
{
	if (!block_dim) error;
	else if (!grid_dim) error;
	else if (block_dim != grid_dim) error;
	_queue.enqueueNDRangeKernel(_kernel, NullRange, grid, block);
}

