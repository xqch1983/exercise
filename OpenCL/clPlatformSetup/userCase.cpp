	bool VarianceFilter::clfilter(const Mat &img)
	{
		//printf("begin using VarianceFilter clFilter*************************\n");

		//char *kernelName = "M:\\OpenTLD\\OpenTLD-master\\OpenTLD-master\\src\\libopentld\\tld\\varianceFilter.cl";
		char *kernelName = "..\\..\\..\\src\\libopentld\\kernel\\varianceFilter_kernel.cpp"; // it is conbined to ensemble classify.

		cl_event events[1];
		string sourceStr;
		status = convertToString(kernelName, sourceStr);
		const char *source = sourceStr.c_str();
		size_t sourceSize[] = { strlen(source) };
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

		/*Step 6: Build program. */
		status = clBuildProgram(program, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);
		//printf("mid using GPU*************************\n");
		if (status != CL_SUCCESS)
		{
			cout << "VarianceFilter Error: Getting platforms!" << endl;
			return false;
		}

		cl_mem oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (TLD_WINDOW_OFFSET_SIZE * numWindows) * sizeof(int), (void *)windowOffsets, NULL);
		cl_mem oclbuffII = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (img.size().width)*(img.size().height) * sizeof(int), (void *)iSumMat.data, NULL);
		cl_mem oclbuffIISqure = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (img.size().width)*(img.size().height) * sizeof(int), (void *)fSqreSumMat.data, NULL);
		cl_mem oclbuffDetectionResultVarious = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->variances, NULL);
		cl_mem oclbuffDetectionResultPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->posteriors, NULL);

		cl_mem oclbuffDetectionwindowFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(int), (void *)detectionResult->windowFlags, NULL);

		/*Step 8: Create kernel object */
		kernel = clCreateKernel(program, "varianceFilter", NULL);

		/*Step 9: Sets Kernel arguments.*/
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&oclbuffWindowsOffset);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&oclbuffII);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&oclbuffIISqure);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oclbuffDetectionResultVarious);
		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oclbuffDetectionResultPosteriors);
		status = clSetKernelArg(kernel, 5, sizeof(int), (void *)&numWindows);
		status = clSetKernelArg(kernel, 6, sizeof(float), (void *)&minVar);
		status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&oclbuffDetectionwindowFlags);
		//status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&TLD_WINDOW_OFFSET_SIZE);

		/*Step 10: Running the kernel.*/
		printf("begore opencl kernel numWindows=%d\n", numWindows);
		size_t global_work_size[1] = { numWindows };
		size_t local_work_size[1] = { 256 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &events[0]);
		if (status != CL_SUCCESS)
		{
			cout << "Error:VarianceFilter EnqueueNDRangeKernel!" << endl;
			//return false;
		}
		status = clWaitForEvents(1, &events[0]);

		if (status != CL_SUCCESS)
		{
			printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");

		}
		//cout << "o" << endl;

		status = clReleaseEvent(events[0]);

		//printf("end using GPU*************************\n");
		//for (int i = 0; i < numWindows; i++)
		//	if(detectionResult->windowFlags[i]==1)
		//	printf("detectionResult[%d] is %d\n", i,detectionResult->windowFlags[i]);

		/*Step 11: Read the cout put back to host memory.*/
		//status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, 12 * sizeof(char), output, 0, NULL, NULL);



		clReleaseKernel(kernel);
		clReleaseProgram(program);

		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbuffII);
		clReleaseMemObject(oclbuffIISqure);
		clReleaseMemObject(oclbuffDetectionResultVarious);
		clReleaseMemObject(oclbuffDetectionResultPosteriors);
		clReleaseMemObject(oclbuffDetectionwindowFlags);
		return true;


	}
} 
