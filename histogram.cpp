#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>
#include <CL/cl.h>

/*unsigned int * histogram(unsigned int *image_data, unsigned int _size) {

	unsigned int *img = image_data;
	unsigned int *ref_histogram_results;
	unsigned int *ptr;

	ref_histogram_results = (unsigned int *)malloc(256 * 3 * sizeof(unsigned int));
	ptr = ref_histogram_results;
	memset (ref_histogram_results, 0x0, 256 * 3 * sizeof(unsigned int));

	// histogram of R
	for (unsigned int i = 0; i < _size; i += 3)
	{
		unsigned int index = img[i];
		ptr[index]++;
	}

	// histogram of G
	ptr += 256;
	for (unsigned int i = 1; i < _size; i += 3)
	{
		unsigned int index = img[i];
		ptr[index]++;
	}

	// histogram of B
	ptr += 256;
	for (unsigned int i = 2; i < _size; i += 3)
	{
		unsigned int index = img[i];
		ptr[index]++;
	}

	return ref_histogram_results;
}
*/
int main(int argc, char const *argv[])
{

	unsigned int * histogram_results;
	unsigned int i=0, a, input_size;
	std::fstream inFile("input", std::ios_base::in);
	std::ofstream outFile("0556562.out", std::ios_base::out);

	inFile >> input_size;
	unsigned int *image = new unsigned int[input_size];
	while( inFile >> a ) {
		image[i++] = a;
	}

	size_t output_size = 256 * 3;
	histogram_results = new unsigned int[output_size]();

	//platform
	cl_uint plat_num = 0;
	clGetPlatformIDs(0, NULL, &plat_num);
	cl_platform_id *platforms = new cl_platform_id[plat_num];
	clGetPlatformIDs(plat_num, platforms, NULL);

	//device
	cl_uint device_num = 0;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);
	cl_device_id *devices = new cl_device_id[device_num];
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, device_num, devices, NULL);

	//content
	cl_int err_num;
	cl_context_properties property[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0};
	cl_context context = clCreateContextFromType(property, CL_DEVICE_TYPE_GPU, NULL, NULL, &err_num);

	//cmd queue
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], NULL, &err_num);

	//memory input & output
	cl_mem mem_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * input_size, &image[0], &err_num);
	cl_mem mem_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * output_size, &histogram_results[0], &err_num);

	//program & kernel
	std::string source_str = "__kernel void histogram(__global unsigned int *src_data, __global unsigned int *dest_data) {int id = get_global_id(0);int rgb = id % 3;int val = src_data[id];atomic_inc(&dest_data[rgb * 256 + val]);}";
	const char *source = source_str.c_str();
	const size_t len = source_str.length();
	cl_program prog = clCreateProgramWithSource(context, 1, &source, &len, &err_num);
	err_num = clBuildProgram(prog, device_num, devices, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(prog, "histogram", &err_num);

	//pass para
	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_in);
	err_num = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_out);

	//execute
	const size_t globalSize[] = {input_size, 0, 0};
	err_num = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);

	//read from device
	err_num = clEnqueueReadBuffer(cmdQueue, mem_out, CL_TRUE, 0, sizeof(unsigned int) * output_size, histogram_results, 0, NULL, NULL);

	//histogram_results = histogram(image, input_size);
	for(unsigned int i = 0; i < 256 * 3; ++i) {
		if (i % 256 == 0 && i != 0)
			outFile << std::endl;
		outFile << histogram_results[i]<< ' ';
	}

	inFile.close();
	outFile.close();

	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(mem_in);
	clReleaseMemObject(mem_out);
	clReleaseProgram(prog);
	clReleaseKernel(kernel);
	clReleaseContext(context);

	return 0;
}
