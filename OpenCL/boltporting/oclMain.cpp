#include <CL/cl.h>
#include <CL/cl.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "radix_sort.hpp"
#include <sys/time.h>
 
using namespace std;


#include <algorithm>
#include "test_utils.hpp"
 
#include <vector>

#define SUCCESS 0
#define FAILURE 1

  
 

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
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
	return FAILURE;
}
//



int main(int argc, char* argv[])
{

	int localSize = 256;
	const int RADIX = 4;
	const int RADICES = (1 << RADIX);
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	// use device[1] because that's a GPU; device[0] is the CPU
	cl::Device default_device = all_devices[0];
	std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
	cl::Context context( default_device );
	cl::CommandQueue queue(context, default_device);



	// get ready for kernel commonKernels
	cl::Program::Sources sources_commonKernels;
	const char *filename_commonKernels = "commonKernels.cpp";
	//const char *filename_commonKernels = "C:\\Users\\Kevin.xie\\Documents\\Visual Studio 2013\\Projects\\axpy-cplusplus-template\\axpy-cplusplus-template\\commonKernels.cl";
	std::string kernel_code_commonKernels;
	convertToString(filename_commonKernels, kernel_code_commonKernels);
	sources_commonKernels.push_back({ kernel_code_commonKernels.c_str(), kernel_code_commonKernels.length() });
	cl::Program program_commonKernels(context, sources_commonKernels);
	if (program_commonKernels.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program_commonKernels.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}
	cl::Kernel histogramAscInstantiated = cl::Kernel(program_commonKernels, "histogramAscInstantiated");
	//cl::Kernel histogramDescInstantiated = cl::Kernel(program_commonKernels, "histogramDescInstantiated");
	//cl::Kernel histogramSignedAscInstantiated = cl::Kernel(program_commonKernels, "histogramSignedAscInstantiated");
	//cl::Kernel histogramSignedDescInstantiated = cl::Kernel(program_commonKernels, "histogramSignedDescInstantiated");
	cl::Kernel scanInstantiated = cl::Kernel(program_commonKernels, "scanInstantiated");
 
  
	// get ready for kernel uintKernels
	cl::Program::Sources sources_uintKernels;	 
	const char *filename_uintKernels = "uintKernels.cpp";
	std::string kernel_code_uintKernels;
	convertToString(filename_uintKernels, kernel_code_uintKernels);
	sources_uintKernels.push_back({ kernel_code_uintKernels.c_str(), kernel_code_uintKernels.length() });
	cl::Program program_uintKernels(context, sources_uintKernels);
	if (program_uintKernels.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program_uintKernels.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}
	cl::Kernel   permuteAscInstantiated = cl::Kernel(program_uintKernels, "permuteAscInstantiated");
	//cl::Kernel   permuteDescInstantiated = cl::Kernel(program_uintKernels, "permuteDescInstantiated");
 
	// get ready for kernel floatKernels
	cl::Program::Sources sources_floatKernels;
	const char *filename_floatKernels = "floatKernels.cpp";
	std::string kernel_code_floatKernels;
	convertToString(filename_floatKernels, kernel_code_floatKernels);
	sources_floatKernels.push_back({ kernel_code_floatKernels.c_str(), kernel_code_floatKernels.length() });
	cl::Program program_floatKernels(context, sources_floatKernels);
	if (program_floatKernels.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program_floatKernels.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}
	cl::Kernel  flipFloatInstantiated = cl::Kernel(program_floatKernels, "flipFloatInstantiated");
	cl::Kernel   inverseFlipFloatInstantiated = cl::Kernel(program_floatKernels, "inverseFlipFloatInstantiated");
	// push back kernels to one kernel std::vector
	std::vector< cl::Kernel > Kernels;
	Kernels.push_back(flipFloatInstantiated);
	Kernels.push_back(inverseFlipFloatInstantiated);
	Kernels.push_back(histogramAscInstantiated);
	Kernels.push_back(scanInstantiated);
	Kernels.push_back(permuteAscInstantiated);


	//get ready for input and output buffer.
	int szElements = 1000000; //2048000; //2^21
	if(argc>2)
	szElements = atoi(argv[1]);
	else 
	szElements = 1000000; 
	int interTimes = 10;
	if (argc > 2)
		interTimes = atoi(argv[2]);
	else
		interTimes = 10;
	int warmUpTime = 3;
	//Timer t; // anoter time method
	 
 

	std::vector<float> boltKey(szElements);
	std::vector<int> boltValue(szElements);
	 

	std::generate(boltKey.begin(), boltKey.end(), test_utils::RandomNumber_float);
	std::generate(boltValue.begin(), boltValue.end(), test_utils::UniqueNumber);
	// set some special values to verify correctiness.
	boltKey[2] = 1000.5;
	boltValue[2] = 2;
	boltKey[5]= -88.8;
	boltValue[5] = 5;
	boltKey[3] = -188.8;
	boltValue[3] = 3;
	  
    // sort on host or CPU side.
	using key_value = std::pair<float, float>;
	using key_type = float;
	using value_type = int;

	std::vector<key_value> expected(szElements);
	for (size_t i = 0; i < szElements; i++)
	{
		expected[i] = key_value(boltKey[i], boltValue[i]);
	}
	bool descending = false;
	unsigned int start_bit = 0;
	unsigned int end_bit = sizeof(key_type) * 8;
 
	sort(expected.begin(), expected.end(), test_utils::cmp1);


	std::vector<key_type> keys_expected(szElements);
	std::vector<value_type> values_expected(szElements);
 

	for (size_t i = 0; i < szElements; i++)
	{
		keys_expected[i] = expected[i].first;
		values_expected[i] = expected[i].second;
	}
	 
 
	cl::Buffer dvInputOutput(context, CL_MEM_READ_WRITE, boltKey.size()* sizeof(float) );
	cl::Buffer dvValueInputOutput(context, CL_MEM_READ_WRITE, boltValue.size()* sizeof(int) );

	cl::Buffer dvSwapInputData(context, CL_MEM_READ_WRITE, szElements * sizeof(float));  
	cl::Buffer dvValueInputOutputSwap(context, CL_MEM_READ_WRITE, szElements * sizeof(float) );

	cl::Buffer dvHistogramBins(context, CL_MEM_READ_WRITE, (localSize * RADICES) * sizeof(float));
 

	// push write commands to queue
	queue.enqueueWriteBuffer(dvInputOutput, CL_TRUE, 0, boltKey.size()* sizeof(float), boltKey.data());
	queue.enqueueWriteBuffer(dvValueInputOutput, CL_TRUE, 0, boltValue.size()* sizeof(int), boltValue.data());
	//warm-up  time	
	for (int j = 0; j <warmUpTime; j++)
		radix_sort_key_fp32_clMem(context, queue, Kernels, boltKey.size(), dvInputOutput, dvSwapInputData, dvHistogramBins, dvValueInputOutput, dvValueInputOutputSwap);

	//t.Start();
	feifei::UnixTimer timer;
	timer.Restart();
	for (int j = 0; j <interTimes; j++)
		radix_sort_key_fp32_clMem(context, queue, Kernels, boltKey.size(), dvInputOutput, dvSwapInputData, dvHistogramBins, dvValueInputOutput, dvValueInputOutputSwap);
	//t.End();
	timer.Stop();
//	double delta = t.GetDelta();
//	std::cout << "elsatp time \t" << delta / interTimes << "ms" << std::endl;
	std::cout << "********interface on Host <<radix_sort_key_fp32_clMem>> elsatp time \t" << timer.ElapsedMilliSec / interTimes << "ms" << std::endl;

	queue.finish();
	queue.enqueueReadBuffer(dvInputOutput, CL_TRUE, 0, boltKey.size()* sizeof(float), boltKey.data());
	queue.enqueueReadBuffer(dvValueInputOutput, CL_TRUE, 0, boltValue.size()* sizeof(float), boltValue.data());
	queue.finish();
 
#if 1
	 
	if (test_utils::is_eq(boltKey, keys_expected))
		std::cout << "Successfully #############sort key is successful\n";
	else 
		std::cout << "ERROR**************sort key is not successful\n";
	
	if (test_utils::is_eq(boltValue, values_expected))
		std::cout << "Successfully #############sort value is successful\n";
	else
		std::cout << "ERROR**************sort value is not successful\n";

#endif  
	
	
#if 0
	test_utils::prinprintOutput(boltKey, boltValue);
#endif
#if 0
	test_utils::prinprintOutput(keys_expected, values_expected);
#endif
 
	//std::cout << "elsatp time \t" << delta / interTimes << "ms" << std::endl;
	std::cout << "elsatp time \t" << timer.ElapsedMilliSec / interTimes << "ms" << std::endl;
	std::cout << "szElements \t" << szElements  << std::endl;
	std::cout << "interTimes \t" << interTimes  << std::endl;
	return 0;
 

}
