// xqc.cpp : Defines the entry point for the console application.
#include  "CL/opencl.h"
 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream> 
#include "initialization.hpp"

#define SUCCESS 0
#define FAILURE 1
#define dbg 1
#define SIZE 1024*30000
//#define SIZE 2560*2560
int gettime()
{
	//timeval tv;
	// gettimeofday(&tv,NULL);
	//	return (int)tv.tv_sec*1000000 + tv.tv_usec ;
	return 0;
}


long int get_time()
{
	//	timespec tp;

	//	 clock_gettime(CLOCK_MONOTONIC, &tp);

	//	return (long int)tp.tv_sec*1000000000 + tp.tv_nsec;
	return 0;
}
using namespace anakin;
using namespace anakin::saber;

int main(int argc, char*argv[])
{
	std::vector<Info> oclinfo;
	int devicetype = OCL_DEVICE_TYPE_GPU;
	int inumdevices = getDevice(oclinfo, devicetype);
	const Context *clcxt = Context::getContext();
	clcxt->setContext(oclinfo[0]);
	const char *filename = "Anakin_kernel.cl";

	cl_kernel kernel_vscopy = openCLGetKernelFromSource_2(clcxt, filename, std::string("vscopy"), NULL);
	/********************************************************************************************************/
	int n;
	/*Step1: Getting platforms and choose an available one.*/
	cl_int	status;
	float *h_a, *point_a;
	float  *h_c, *point_c;
	//int size=SIZE;
	if (argc<2)
		printf("*****Pls input the size number********:\n");
	//int xqc
	//int size = atoi(argv[1]);
	int size = 10240;
	//int localsize = atoi(argv[2]);
	int localsize = 256;
	//int vector = atoi(argv[3]);
	int vector = 2;
	//n = atoi(argv[4]);
	n = 1;
	h_a = (float*)malloc(size*sizeof(float));
	h_c = (float*)malloc(size*sizeof(float));
	point_a = h_a;
	point_c = h_c;
	for (int i = 0; i<size; i++)
	{
		h_a[i] = (i + 1)*1.0;
		//	cout<<h_a[i]<<"\t"<<endl;
	}
	cl_mem d_a = clCreateBuffer(clcxt->impl->clContext, CL_MEM_READ_ONLY, (size)* sizeof(float), NULL, NULL);
	cl_mem d_c = clCreateBuffer(clcxt->impl->clContext, CL_MEM_WRITE_ONLY, (size)* sizeof(float), NULL, NULL);

	/*Step 8: Create kernel object */
	/*************************************** test scopy  ********************************************/
	status = clSetKernelArg(kernel_vscopy, 0, sizeof(cl_mem), (void *)&d_a);
	status = clSetKernelArg(kernel_vscopy, 1, sizeof(cl_mem), (void *)&d_c);

	if (status != CL_SUCCESS)
		printf("erro when clSetKernelArgn\n");
	status = clEnqueueWriteBuffer(clcxt->impl->clCmdQueue, d_a, CL_TRUE, 0, size*sizeof(float), h_a, 0, NULL, NULL);
 	/*Step 10: Running the kernel.*/
	size_t global_work_size[1] = { size / (vector*n) };
	size_t local_work_size[1] = { localsize };
	status = clEnqueueNDRangeKernel(clcxt->impl->clCmdQueue, kernel_vscopy, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFinish(clcxt->impl->clCmdQueue);
	if (status != CL_SUCCESS)
		printf("#0 ******************scopy passed when NDRangeKernel\n");
	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(clcxt->impl->clCmdQueue, d_c, CL_TRUE, 0, size* sizeof(float), h_c, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		printf("erro when Readbuff\n");
	printf("after scopy op:\n");
	cl_kernel kernel_vsadd = openCLGetKernelFromSource_2(clcxt, filename, std::string("vsadd"), NULL);
	cl_kernel kernel_vssub = openCLGetKernelFromSource_2(clcxt, filename, std::string("vssub"), NULL);
	/*************************************** test sadd ********************************************/
	status = clSetKernelArg(kernel_vsadd, 0, sizeof(cl_mem), (void *)&d_a);
	status = clSetKernelArg(kernel_vsadd, 1, sizeof(cl_mem), (void *)&d_c);
	status = clEnqueueWriteBuffer(clcxt->impl->clCmdQueue, d_a, CL_TRUE, 0, size*sizeof(float), h_a, 0, NULL, NULL);
	/*Step 10: Running the kernel.*/
	 
	status = clEnqueueNDRangeKernel(clcxt->impl->clCmdQueue, kernel_vsadd, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFinish(clcxt->impl->clCmdQueue);
	if (status != CL_SUCCESS)
		printf("#1 *****************sadd passed when NDRangeKernel\n");
	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(clcxt->impl->clCmdQueue, d_c, CL_TRUE, 0, size* sizeof(float), h_c, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		printf("erro when Readbuff\n");
	printf("after sadd  op:\n");
	cl_kernel kernel_vscopy2 = openCLGetKernelFromSource_2(clcxt, filename, std::string("vscopy"), NULL);
	/*************************************** test scopy #2  ********************************************/
	status = clSetKernelArg(kernel_vscopy2, 0, sizeof(cl_mem), (void *)&d_a);
	status = clSetKernelArg(kernel_vscopy2, 1, sizeof(cl_mem), (void *)&d_c);
	status = clEnqueueWriteBuffer(clcxt->impl->clCmdQueue, d_a, CL_TRUE, 0, size*sizeof(float), h_a, 0, NULL, NULL);
	/*Step 10: Running the kernel.*/
	 
	status = clEnqueueNDRangeKernel(clcxt->impl->clCmdQueue, kernel_vscopy2, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFinish(clcxt->impl->clCmdQueue);
	if (status != CL_SUCCESS)
		printf("*****************scopy passed when NDRangeKernel\n");
	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(clcxt->impl->clCmdQueue, d_c, CL_TRUE, 0, size* sizeof(float), h_c, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		printf("erro when Readbuff\n");
	printf("#2 after scopy  op:\n");
	
	
	
	/*Step 12: Clean the resources.*/
	status = clReleaseMemObject(d_a);		
	status = clReleaseMemObject(d_c);
	std::cout << "Passed!\n";
	free(point_a);
	free(point_c);
	return SUCCESS;
}

