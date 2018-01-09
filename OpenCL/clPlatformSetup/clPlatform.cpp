/*
Editior: xieqingchun 
e-mail: xieqingchun@126.com
Describtion: one class for campsulate the interface for OpenCL.
*/
#include "stdafx.h"
#include "platform.h"
//#pragma warning( disable : 4996 )

#include <opencv/cv.h>
platform_GPU::platform_GPU()
{

	oclSetup();

}

int platform_GPU::oclSetup()
{
	platform = NULL;	//the chosen platform
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout << "DetectorCascade oclSetup Error: Getting platforms!" << endl;
		return 0;
	}

	/*For clarity, choose the first available platform. */
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}
	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	numDevices = 0;

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	cout << "numDevices::" << numDevices << endl;
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}
	for (cl_uint i = 0; i < numDevices; ++i)
	{
		char deviceName[1024];
		status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceName),
			deviceName, NULL);

		std::cout << "Device " << i << " : " << deviceName
			<< " Device ID is " << devices[i] << std::endl;
	}

	/*Step 3: Create context.*/
	context = clCreateContext(NULL, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);

	/*Step 4: Creating command queue associate with the context.*/
	commandQueue = clCreateCommandQueueWithProperties(context, devices[SELECTED_DEVICE_ID], NULL, NULL);

	/*Step 5: Create program object */
	//  const char *kernelName = "M:\\OpenTLD\\OpenTLD-master\\OpenTLD-master\\src\\opentld\\HelloWorld_Kernel.cl";
	const char *kernelName = "C:\\Users\\Kevin.xie\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\ConsoleApplication1\\kernel\\varianceFilter_kernel.cpp";


	string sourceStr;
	status = convertToString(kernelName, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

	/*Step 6: Build program. */
	status = clBuildProgram(program, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t log_size;
		char* program_log;
		clGetProgramBuildInfo(program, devices[SELECTED_DEVICE_ID], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		clGetProgramBuildInfo(program, devices[SELECTED_DEVICE_ID], CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);


		cout << "DetectorCascade::oclBuildKernelError:clBuildProgram  !" << endl;
		return false;
	}

	//step 7 CreateKernel

	kernel_cube2ERP = clCreateKernel(program, "vecadd", NULL);

	//(matImg.cols) * (matImg.rows)
	//cl_mem srcdata = clCreateBuffer(context, CL_MEM_READ_ONLY, (1624) * (1624) * sizeof(char), NULL, NULL);
	//cl_mem tsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (1624) * (1624)* sizeof(int), NULL, NULL);

	//?--------------------------10.ÔËÐÐÄÚºË---------------------------------??



}

int platform_GPU::convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return false;
}

platform_GPU::~platform_GPU()
{
	/*Step 12: Clean the resources.*/
	if (kernel_cube2ERP != NULL)
		status = clReleaseKernel(kernel_cube2ERP);				//Release kernel.
	status = clReleaseProgram(program);				//Release the program object.
	status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
	status = clReleaseContext(context);				//Release context.
	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}





}