#pragma once
#include  <CL/cl.h>
#include <iostream>
#include<fstream>
using namespace std;
#define SELECTED_DEVICE_ID 0
class platform_GPU
{
public:
	platform_GPU();
	int oclSetup();
	int convertToString(const char *filename, std::string& s);

	~platform_GPU();
public:
	/*Step1: Getting platforms and choose an available one.*/
	cl_device_id    *devices;
	cl_context       context;
	cl_command_queue commandQueue;
	cl_program       program;
	cl_kernel        kernel_cube2ERP;

private:
	cl_int	        status;
	cl_uint		    numDevices;
	cl_uint         numPlatforms;	//the NO. of platforms
	cl_platform_id  platform;	//the chosen platform
	cl_event		events[1];
};