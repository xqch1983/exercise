#include <CL/cl.h>
#include <CL/cl.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#include <sys/time.h>
using namespace std; 
#include <iostream>
#include <ctime>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include "ff_timer.h"
#define printTime 1
int radix_sort_key_fp32_clMem(cl::Context context, cl::CommandQueue queue, vector< cl::Kernel > Kernels, uint szElements,  cl::Buffer &dvInputOutput, cl::Buffer &dvSwapInputData, cl::Buffer &dvHistogramBins,cl::Buffer &dvValueInputOutput, cl::Buffer &dvValueInputOutputSwap)
{
	  
	cl_int l_Error = CL_SUCCESS;
	feifei::UnixTimer timer;
	const int RADIX = 4;
	const int RADICES = (1 << RADIX); //Values handeled by each work-item?
	int computeUnits = 32;
	//	if (computeUnits > 32)
	//	computeUnits = 64;

	int localSize = 256;
	int wavefronts = 8;
	int numGroups = computeUnits * wavefronts; // 64*8 = 512

	::cl::Buffer   clInputData = dvInputOutput;
	::cl::Buffer clSwapData = dvSwapInputData;
	::cl::Buffer   clInputDataValue = dvValueInputOutput;
	::cl::Buffer   clSwapDataValue =  dvValueInputOutputSwap;
	::cl::Buffer clHistData = dvHistogramBins;


	::cl::Kernel flipFloatKernel;
	::cl::Kernel inverseFlipFloatKernel;

	flipFloatKernel = Kernels[0];
	inverseFlipFloatKernel = Kernels[1];


	::cl::Kernel histKernel = Kernels[2];
	::cl::Kernel scanLocalKernel = Kernels[3];
	::cl::Kernel permuteKernel = Kernels[4];

	int swap = 0;
	const int ELEMENTS_PER_WORK_ITEM = 4;
	int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);//set at 1024
	int nBlocks = (int)(szElements + blockSize - 1) / (blockSize);
	struct b3ConstData
	{
		int m_n;
		int m_nWGs;
		int m_startBit;
		int m_nBlocksPerWG;
	};
	b3ConstData cdata;
	cdata.m_n = (int)szElements;
	cdata.m_nWGs = (int)numGroups;
	cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1) / numGroups;
	if (nBlocks < numGroups)
	{
		cdata.m_nBlocksPerWG = 1;
		numGroups = nBlocks;
		cdata.m_nWGs = numGroups;
	}

	cl_int res = histKernel.setArg(1, clHistData);
	res = scanLocalKernel.setArg(0, clHistData);
	if (res != CL_SUCCESS)
		printf("\t%s,%d,%s\n", __FILE__, __LINE__, "scanLocalKernel 0.setArg");
	scanLocalKernel.setArg(1, (int)numGroups);
	if (res != CL_SUCCESS)
		printf("\t%s,%d,%s\n", __FILE__, __LINE__, "scanLocalKernel 1 .setArg");
	res = scanLocalKernel.setArg(2, localSize * 2 * sizeof(float), NULL);
	if (res != CL_SUCCESS)
		printf("\t%s,%d,%s\n", __FILE__, __LINE__, "scanLocalKernel 2.setArg");
	res = permuteKernel.setArg(1, clHistData);
	if (res != CL_SUCCESS)
		printf("\t%s,%d,%s\n", __FILE__, __LINE__, "permuteKernel 1.setArg");
	res = flipFloatKernel.setArg(0, clInputData);
	if (res != CL_SUCCESS)
		printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 0.setArg");

	res = flipFloatKernel.setArg(1, cdata);
	if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");

#ifdef printTime
	timer.Restart();
#endif
	l_Error = queue.enqueueNDRangeKernel(
		flipFloatKernel,
		::cl::NullRange,
		::cl::NDRange(numGroups*localSize),
		::cl::NDRange(localSize), //This mul will be removed when permute is optimized
		NULL,
		NULL);

	for (int bits = 0; bits < (sizeof(float) * 8); bits += RADIX)
	{
		cdata.m_startBit = bits;
		res = histKernel.setArg(2, cdata);
		if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
		if (swap == 0)
		{
			res = histKernel.setArg(0, clInputData);
			if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
		}

		else
		{
			res = histKernel.setArg(0, clSwapData);
			if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");


		}
		l_Error = queue.enqueueNDRangeKernel(
			histKernel,
			::cl::NullRange,
			::cl::NDRange(numGroups*localSize),
			::cl::NDRange(localSize), //This mul will be removed when permute is optimized
			NULL,
			NULL);
		l_Error = queue.enqueueNDRangeKernel(
			scanLocalKernel,
			::cl::NullRange,
			::cl::NDRange(localSize),
			::cl::NDRange(localSize), //This mul will be removed when permute is optimized
			NULL,
			NULL);

		res = permuteKernel.setArg(3, cdata);
		if (swap == 0)
		{
			permuteKernel.setArg(0, clInputData);
			permuteKernel.setArg(4, clInputDataValue);
			if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
			permuteKernel.setArg(2, clSwapData);
			permuteKernel.setArg(5, clSwapDataValue);
			if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
		}
		else
		{
			permuteKernel.setArg(0, clSwapData);
			permuteKernel.setArg(4, clSwapDataValue);
			if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
			permuteKernel.setArg(2, clInputData); if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
			permuteKernel.setArg(5, clInputDataValue);
		}
		l_Error = queue.enqueueNDRangeKernel(
			permuteKernel,
			::cl::NullRange,
			::cl::NDRange(numGroups*localSize),
			::cl::NDRange(localSize),
			NULL,
			NULL);
		swap = swap ? 0 : 1;
	}
	inverseFlipFloatKernel.setArg(0, clInputData);
		if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");

	inverseFlipFloatKernel.setArg(1, cdata);
		if (res != CL_SUCCESS) 	 	printf("\t%s,%d,%s\n", __FILE__, __LINE__, "flipFloatKernel 1.setArg");
	l_Error = queue.enqueueNDRangeKernel(
		inverseFlipFloatKernel,
		::cl::NullRange,
		::cl::NDRange(numGroups*localSize),
		::cl::NDRange(localSize), //This mul will be removed when permute is optimized
		NULL,
		NULL);
	queue.finish();
#ifdef printTime
	timer.Stop();
	std::cout << ".................dispatch kernels in host spend time \t" << timer.ElapsedMilliSec  << "ms" << std::endl;
#endif
	return 0;

}


