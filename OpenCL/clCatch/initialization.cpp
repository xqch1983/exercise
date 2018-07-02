
#include "initialization.hpp"
#include <iomanip>
#include <fstream>
#include <mutex>
#include <assert.h>
#include <cstring>
using namespace std;
using std::cout;
using std::endl;

//#define PRINT_KERNEL_RUN_TIME
#define RUN_TIMES 100
#ifndef CL_MEM_USE_PERSISTENT_MEM_AMD
#define CL_MEM_USE_PERSISTENT_MEM_AMD 0
#endif


namespace anakin
{
	namespace saber 
	{

		/*
		 * The binary caching system to eliminate redundant program source compilation.
		 * Strictly, this is not a cache because we do not implement evictions right now.
		 * We shall add such features to trade-off memory consumption and performance when necessary.
		 */
		auto_ptr<Context> Context::clCxt;
		Context *clCxt = NULL;	
		int Context::val = 0;

		auto_ptr<ProgramCache> ProgramCache::programCache;
		ProgramCache *programCache = NULL;

		auto_ptr<KernelCache> KernelCache::kernelCache;
		KernelCache *kernelCache = NULL;

		DevMemType gDeviceMemType = DEVICE_MEM_DEFAULT;
		DevMemRW gDeviceMemRW = DEVICE_MEM_R_W;
		
		int gDevMemTypeValueMap[5] = {0, 
			CL_MEM_ALLOC_HOST_PTR,
			CL_MEM_USE_HOST_PTR,
			CL_MEM_COPY_HOST_PTR,
			CL_MEM_USE_PERSISTENT_MEM_AMD};
		int gDevMemRWValueMap[3] = {CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
		/********************************  ProgramCache definition    ************************************/
		ProgramCache::ProgramCache()
		{
			codeCache.clear();
			cacheSize = 0;
		}

		ProgramCache::~ProgramCache()
		{
			releaseProgram();
		}

		cl_program ProgramCache::progLookup(string srcsign)
		{
			map<string, cl_program>::iterator iter;
			iter = codeCache.find(srcsign);
			if(iter != codeCache.end())
				return iter->second;
			else
				return NULL;
		}

		void ProgramCache::addProgram(string srcsign , cl_program program)
		{
			if(!progLookup(srcsign))
			{
				codeCache.insert(map<string, cl_program>::value_type(srcsign, program));
			}
		}

		void ProgramCache::releaseProgram()
		{
			map<string, cl_program>::iterator iter;
			for(iter = codeCache.begin(); iter != codeCache.end(); iter++)
			{
				//openCLSafeCall(clReleaseProgram(iter->second));
				clReleaseProgram(iter->second);
			}
			codeCache.clear();
			cacheSize = 0;
		}
		/********************************  KernelCache definition    ************************************/
		KernelCache::KernelCache()
		{
			mapKernelCache.clear();
			cachKernelSize = 0;
		}

		KernelCache::~KernelCache()
		{
			releaseKernel();
		}


		cl_kernel KernelCache::kernelLookup(std::pair<std::string, std::string> srcsign)
		{
			map<std::pair<std::string, std::string>, cl_kernel>::iterator iter;
			iter = mapKernelCache.find(srcsign);
			if (iter != mapKernelCache.end())
				return iter->second;
			else
				return NULL;
		}
		void KernelCache::addKernel(std::pair<std::string, std::string> srcsign, cl_kernel kernel)
		{
			if (!kernelLookup(srcsign))
			{
				mapKernelCache.insert(map<std::pair<std::string, std::string>, cl_kernel>::value_type(srcsign, kernel));
			}
		}
		void KernelCache::releaseKernel()
		{
			map<std::pair<std::string, std::string>, cl_kernel>::iterator iter;
			for (iter = mapKernelCache.begin(); iter != mapKernelCache.end(); iter++)
			{
				//openCLSafeCall(clReleaseProgram(iter->second));
				clReleaseKernel(iter->second);
			}
			mapKernelCache.clear();
			cachKernelSize = 0;
		} 

		////////////////////////Common OpenCL specific calls///////////////
		int getDevMemType(DevMemRW& rw_type, DevMemType& mem_type)
		{ 
			rw_type = gDeviceMemRW; 
			mem_type = gDeviceMemType; 
			return Context::getContext()->impl->unified_memory;
		}

		int setDevMemType(DevMemRW rw_type, DevMemType mem_type)
		{ 
			if( (mem_type == DEVICE_MEM_PM && Context::getContext()->impl->unified_memory == 0) ||
					mem_type == DEVICE_MEM_UHP ||
					mem_type == DEVICE_MEM_CHP )
				return -1;
			gDeviceMemRW = rw_type;
			gDeviceMemType = mem_type;
			return 0; 
		}

		struct Info::Impl
		{
			cl_platform_id oclplatform;
			std::vector<cl_device_id> devices;
			std::vector<std::string> devName;

			cl_context oclcontext;
			cl_command_queue clCmdQueue;
			int devnum;
			cl_uint maxDimensions;
			size_t maxWorkGroupSize;
			size_t *maxWorkItemSizes;
			cl_uint maxComputeUnits;
			char extra_options[512];
			int  double_support;
			Impl()
			{
				memset(extra_options, 0, 512);
			}
		};

		inline int divUp(int total, int grain)
		{
			return (total + grain - 1) / grain;
		}

		int getDevice(std::vector<Info> &oclinfo, int devicetype)
		{
			switch(devicetype)
			{
				case OCL_DEVICE_TYPE_DEFAULT:
				case OCL_DEVICE_TYPE_CPU:
				case OCL_DEVICE_TYPE_GPU:
				case OCL_DEVICE_TYPE_ACCELERATOR:
				case OCL_DEVICE_TYPE_ALL:
					break;
				default:
					//xqc CV_Error(CV_GpuApiCallError, "Unkown device type");
					printf("Unkown device type");
			}
			int devcienums = 0;
			// Platform info
			cl_int status = 0;
			cl_uint numPlatforms;
			Info ocltmpinfo;
			clGetPlatformIDs(0, NULL, &numPlatforms);
			// openCLSafeCall(clGetPlatformIDs(0, NULL, &numPlatforms));
			// CV_Assert(numPlatforms > 0);xqc
			cl_platform_id *platforms = new cl_platform_id[numPlatforms];
			clGetPlatformIDs(numPlatforms, platforms, NULL);
			// openCLSafeCall(clGetPlatformIDs(numPlatforms, platforms, NULL));
			char deviceName[256];
			for (unsigned i = 0; i < numPlatforms; ++i)
			{
				cl_uint numsdev;
				status = clGetDeviceIDs(platforms[i], devicetype, 0, NULL, &numsdev);
				if(status != CL_DEVICE_NOT_FOUND)
				{
					//openCLVerifyCall(status);

				}
				if(numsdev > 0)
				{
					devcienums += numsdev;
					cl_device_id *devices = new cl_device_id[numsdev];
					clGetDeviceIDs(platforms[i], devicetype, numsdev, devices, NULL);
					//openCLSafeCall(clGetDeviceIDs(platforms[i], devicetype, numsdev, devices, NULL));xqc
					ocltmpinfo.impl->oclplatform = platforms[i];
					for(unsigned j = 0; j < numsdev; j++)
					{
						ocltmpinfo.impl->devices.push_back(devices[j]);
						clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, deviceName, NULL);
						//openCLSafeCall(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, deviceName, NULL));xqc
						ocltmpinfo.impl->devName.push_back(std::string(deviceName));
						ocltmpinfo.DeviceName.push_back(std::string(deviceName));
					}
					delete[] devices;
					oclinfo.push_back(ocltmpinfo);
					ocltmpinfo.release();
				}
			}
			delete[] platforms;
			if(devcienums > 0)
			{
				setDevice(oclinfo[0]);
			}
			return devcienums;
		}

		static void fillClcontext(Info &oclinfo)
		{
			//get device information
			size_t devnum = oclinfo.impl->devnum;

			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_GROUP_SIZE,
					sizeof(size_t), (void *)&oclinfo.impl->maxWorkGroupSize, NULL);
			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_GROUP_SIZE,
					sizeof(size_t), (void *)&oclinfo.impl->maxWorkGroupSize, NULL) ;
			//openCLSafeCall(clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_GROUP_SIZE,
			//	sizeof(size_t), (void *)&oclinfo.impl->maxWorkGroupSize, NULL));
			//openCLSafeCall(clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			//                              sizeof(cl_uint), (void *)&oclinfo.impl->maxDimensions, NULL));
			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
					sizeof(cl_uint), (void *)&oclinfo.impl->maxDimensions, NULL);

			oclinfo.impl->maxWorkItemSizes = new size_t[oclinfo.impl->maxDimensions];
			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_ITEM_SIZES,
					sizeof(size_t)*oclinfo.impl->maxDimensions, (void *)oclinfo.impl->maxWorkItemSizes, NULL);
			//openCLSafeCall(clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_WORK_ITEM_SIZES,
			//	sizeof(size_t)*oclinfo.impl->maxDimensions, (void *)oclinfo.impl->maxWorkItemSizes, NULL));
			//openCLSafeCall(clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_COMPUTE_UNITS,
			//                             sizeof(cl_uint), (void *)&oclinfo.impl->maxComputeUnits, NULL));
			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_MAX_COMPUTE_UNITS,
					sizeof(cl_uint), (void *)&oclinfo.impl->maxComputeUnits, NULL);
			//initialize extra options for compilation. Currently only fp64 is included.
			//Assume 4KB is enough to store all possible extensions.

			const int EXT_LEN = 4096 + 1 ;
			char extends_set[EXT_LEN];
			size_t extends_size;
			// openCLSafeCall(clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_EXTENSIONS,
			//                               EXT_LEN, (void *)extends_set, &extends_size));
			clGetDeviceInfo(oclinfo.impl->devices[devnum], CL_DEVICE_EXTENSIONS,
					EXT_LEN, (void *)extends_set, &extends_size);
			// CV_Assert(extends_size < (size_t)EXT_LEN);
			extends_set[EXT_LEN - 1] = 0;
			memset(oclinfo.impl->extra_options, 0, 512);
			oclinfo.impl->double_support = 0;
			int fp64_khr = string(extends_set).find("cl_khr_fp64");

			if(fp64_khr >= 0 && fp64_khr < EXT_LEN)
			{
				sprintf(oclinfo.impl->extra_options , "-D DOUBLE_SUPPORT");
				oclinfo.impl -> double_support = 1;
			}
			Context::setContext(oclinfo);

		}

		void setDevice(Info &oclinfo, int devnum)
		{
			//CV_Assert(devnum >= 0);
			//assert(devnum >= 0);
			cl_int status = 0;
			cl_context_properties cps[3] =
			{
				CL_CONTEXT_PLATFORM, (cl_context_properties)(oclinfo.impl->oclplatform), 0
			};
			oclinfo.impl->devnum = devnum;
			oclinfo.impl->oclcontext = clCreateContext(cps, 1, &oclinfo.impl->devices[devnum], NULL, NULL, &status);
			//openCLVerifyCall(status);xqc
			//create the command queue using the first device of the list
			oclinfo.impl->clCmdQueue = clCreateCommandQueue(oclinfo.impl->oclcontext, oclinfo.impl->devices[devnum],
					CL_QUEUE_PROFILING_ENABLE, &status);
			//openCLVerifyCall(status);
			fillClcontext(oclinfo);
		}

		void setDeviceEx(Info &oclinfo, void *ctx, void *q, int devnum)
		{
			// CV_Assert(devnum >= 0);
			oclinfo.impl->devnum = devnum;
			if(ctx && q)
			{
				oclinfo.impl->oclcontext = (cl_context)ctx;
				oclinfo.impl->clCmdQueue = (cl_command_queue)q;
				clRetainContext((cl_context)ctx);
				clRetainCommandQueue((cl_command_queue)q);
				fillClcontext(oclinfo);
			}
		}

		void *getoclContext()
		{
			return &(Context::getContext()->impl->clContext);
		}

		void *getoclCommandQueue()
		{
			return &(Context::getContext()->impl->clCmdQueue);
		}


		void setBinpath(const char *path)
		{
			Context *clcxt = Context::getContext();
			clcxt->impl->Binpath = path;
		} 

		int savetofile(const Context*,  cl_program &program, const char *fileName)
		{
			size_t binarySize;
			/*openCLSafeCall(clGetProgramInfo(program,
			  CL_PROGRAM_BINARY_SIZES,
			  sizeof(size_t),
			  &binarySize, NULL));*/
			clGetProgramInfo(program,
					CL_PROGRAM_BINARY_SIZES,
					sizeof(size_t),
					&binarySize, NULL);
			char* binary = (char*)malloc(binarySize);
			if(binary == NULL)
			{
				// CV_Error(CV_StsNoMem, "Failed to allocate host memory.");
			}
			/*openCLSafeCall(clGetProgramInfo(program,
			  CL_PROGRAM_BINARIES,
			  sizeof(char *),
			  &binary,
			  NULL));*/
			clGetProgramInfo(program,
					CL_PROGRAM_BINARIES,
					sizeof(char *),
					&binary,
					NULL);

			FILE *fp = fopen(fileName, "wb+");
			if(fp != NULL)
			{
				fwrite(binary, binarySize, 1, fp);
				free(binary);
				fclose(fp);
			}
			return 1;
		}





		cl_kernel openCLGetKernelFromSource(const Context *clCxt, const char **source, string kernelName,const char *build_options)
		{
			cl_kernel kernel;
			cl_program program ;
			cl_int status = 0;
			stringstream src_sign;
			string srcsign;
			string filename;
			//CV_Assert(programCache != NULL);xqc

			if(NULL != build_options)
			{
				src_sign << (int64)(*source) << clCxt->impl->clContext << "_" << build_options;
			}
			else
			{
				src_sign << (int64)(*source) << clCxt->impl->clContext;
				//src_sign << (int64)(*source) ;
			}
			srcsign = src_sign.str();
			//srcsign = "Anakin";//xqc
			program = NULL;
			program = programCache->progLookup(srcsign);

			if(!program)
			{
				//config build programs
				char all_build_options[1024];
				memset(all_build_options, 0, 1024);
				char zeromem[512] = {0};
				if(0 != memcmp(clCxt -> impl->extra_options, zeromem, 512))
					strcat(all_build_options, clCxt -> impl->extra_options);
				strcat(all_build_options, " ");
				if(build_options != NULL)
					strcat(all_build_options, build_options);
				if(all_build_options != NULL)
				{
					filename = clCxt->impl->Binpath  + kernelName + "_" + clCxt->impl->devName + all_build_options + ".clb";
				}
				else
				{
					filename = clCxt->impl->Binpath  + kernelName + "_" + clCxt->impl->devName + ".clb";
				}

				FILE *fp = fopen(filename.c_str(), "rb");
				if(fp == NULL || clCxt->impl->Binpath.size() == 0)    //we should generate a binary file for the first time.
				{
					if(fp != NULL)
						fclose(fp);

					program = clCreateProgramWithSource(
							clCxt->impl->clContext, 1, source, NULL, &status);
					//openCLVerifyCall(status);
					status = clBuildProgram(program, 1, &(clCxt->impl->devices), all_build_options, NULL, NULL);

					if(status == CL_SUCCESS && clCxt->impl->Binpath.size())
						savetofile(clCxt, program, filename.c_str());
				}
				else
				{
					fseek(fp, 0, SEEK_END);
					size_t binarySize = ftell(fp);
					fseek(fp, 0, SEEK_SET);
					char *binary = new char[binarySize];
					fread(binary, binarySize, 1, fp);
					//CV_Assert(1 == fread(binary, binarySize, 1, fp));
					fclose(fp);
					cl_int status = 0;
					program = clCreateProgramWithBinary(clCxt->impl->clContext,
							1,
							&(clCxt->impl->devices),
							(const size_t *)&binarySize,
							(const unsigned char **)&binary,
							NULL,
							&status);
					// openCLVerifyCall(status);
					status = clBuildProgram(program, 1, &(clCxt->impl->devices), all_build_options, NULL, NULL);
					delete[] binary;
				}

				if(status != CL_SUCCESS)
				{
					if(status == CL_BUILD_PROGRAM_FAILURE)
					{
						cl_int logStatus;
						char *buildLog = NULL;
						size_t buildLogSize = 0;
						logStatus = clGetProgramBuildInfo(program,
								clCxt->impl->devices, CL_PROGRAM_BUILD_LOG, buildLogSize,
								buildLog, &buildLogSize);
						if(logStatus != CL_SUCCESS)
							cout << "Failed to build the program and get the build info." << endl;
						buildLog = new char[buildLogSize];
						//CV_DbgAssert(!!buildLog);
						memset(buildLog, 0, buildLogSize);
						//openCLSafeCall(clGetProgramBuildInfo(program, clCxt->impl->devices,     CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL));
						clGetProgramBuildInfo(program, clCxt->impl->devices, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
						cout << "\n\t\t\tBUILD LOG\n";
						cout << buildLog << endl;
						delete [] buildLog;
					}
					//openCLVerifyCall(status);
				}
				//Cache the binary for future use if build_options is null
				if( (programCache->cacheSize += 1) < programCache->MAX_PROG_CACHE_SIZE)
					programCache->addProgram(srcsign, program);
				else
					cout << "Warning: code cache has been full.\n";
			}
			kernel = clCreateKernel(program, kernelName.c_str(), &status);
			//openCLVerifyCall(status);
			return kernel;
		}
		cl_kernel openCLGetKernelFromSource_2(const Context *clCxt, const char *sourceFile, string kernelName, const char *build_options)
		{
			cl_kernel kernel;
			cl_program program;
			cl_int status = 0;
			stringstream src_sign;
			string srcsign;
			string filename;
			typedef pair<string, string> author;
			//CV_Assert(programCache != NULL);xqc
			assert(programCache != NULL);
			//get the static pointers
			//Context *ctx = Context::getContext();
			//ctx->getContext();
			ProgramCache *programCache = ProgramCache::getProgramCache();

			KernelCache *kernelCache = KernelCache::getKernelCache();

			//srcsign = "Anakin";//xqc
			program = NULL;
			program = programCache->progLookup(sourceFile);
			if (program != NULL)
			{
				KernelCache *kernelCache = KernelCache::getKernelCache();
				author temp(sourceFile, kernelName);
				kernel = kernelCache->kernelLookup(temp);
				if (kernel != NULL)
					return kernel;
				else
				{
					kernel = clCreateKernel(program, (const char*)kernelName.c_str(), NULL);
					//if (status == CL_SUCCESS && clCxt->impl->Binpath.size())
					if (status == CL_SUCCESS)
					{
						if ((kernelCache->cachKernelSize += 1) < kernelCache->MAX_KERNEL_CACHE_SIZE)
							kernelCache->addKernel(std::make_pair(sourceFile, kernelName), kernel);
						else
							cout << "Warning: kernel cache has been full.\n";
					}

					return kernel;
				}

			} 
			else // (!program)
			{
				//config build programs
				char all_build_options[1024];
				memset(all_build_options, 0, 1024);
				char zeromem[512] = { 0 };
				if (0 != memcmp(clCxt->impl->extra_options, zeromem, 512))
					strcat(all_build_options, clCxt->impl->extra_options);
				strcat(all_build_options, " ");
				if (build_options != NULL)
					strcat(all_build_options, build_options);

				//FILE *fp = fopen(sourceFile, "rb");
				//assert(fp != NULL); //if( soureFile_type == 0)    //we should generate a binary file for the first time.

				string sourceStr;
				status = convertToString(sourceFile, sourceStr);
				assert(status == 0);

				const char *source = sourceStr.c_str();
				size_t sourceSize[] = { strlen(source) };
				cl_program program = clCreateProgramWithSource(clCxt->impl->clContext, 1, &source, sourceSize, NULL);

				assert(status == CL_SUCCESS);
				//openCLVerifyCall(status);
				status = clBuildProgram(program, 1, &(clCxt->impl->devices), all_build_options, NULL, NULL);
				//if (status == CL_SUCCESS && clCxt->impl->Binpath.size())
				if (status == CL_SUCCESS)
				{
					savetofile(clCxt, program, filename.c_str());
					if ((programCache->cacheSize += 1) < programCache->MAX_PROG_CACHE_SIZE)
						programCache->addProgram(sourceFile, program);
					else
						cout << "Warning: program code cache has been full.\n";
				}

				kernel = clCreateKernel(program, (const char*)kernelName.c_str(), NULL);
				//if (status == CL_SUCCESS && clCxt->impl->Binpath.size())
				if (status == CL_SUCCESS )
				{					 
					if ((kernelCache->cachKernelSize += 1) < kernelCache->MAX_KERNEL_CACHE_SIZE)
						kernelCache->addKernel(std::make_pair(sourceFile, kernelName), kernel);
					else
						cout << "Warning: kernel cache has been full.\n";
				}

				return kernel;
			}


		}


		cl_kernel openCLGetKernelFromSource(const Context * clCxt, const char ** source, string kernelName)
		{
			return   openCLGetKernelFromSource(clCxt, source, kernelName, NULL);
		}



		void openCLVerifyKernel(const Context *clCxt, cl_kernel kernel, size_t *localThreads)
		{
			size_t kernelWorkGroupSize;
			clGetKernelWorkGroupInfo(kernel, clCxt->impl->devices,
					CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0);
			/* openCLSafeCall(clGetKernelWorkGroupInfo(kernel, clCxt->impl->devices,
			   CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0));  
			   CV_Assert( (localThreads[0] <= clCxt->impl->maxWorkItemSizes[0]) &&
			   (localThreads[1] <= clCxt->impl->maxWorkItemSizes[1]) &&
			   (localThreads[2] <= clCxt->impl->maxWorkItemSizes[2]) &&
			   ((localThreads[0] * localThreads[1] * localThreads[2]) <= kernelWorkGroupSize) &&
			   (localThreads[0] * localThreads[1] * localThreads[2]) <= clCxt->impl->maxWorkGroupSize); */ //xqc
		}

		void openCLExecuteKernel(Context * clCxt, const char ** source, string kernelName, vector<std::pair<size_t, const void*>>& args, int globalcols, int globalrows, size_t blockSize, int kernel_expand_depth, int kernel_expand_channel)
		{
		}

#ifdef PRINT_KERNEL_RUN_TIME
		static double total_execute_time = 0;
		static double total_kernel_time = 0;
#endif
		void openCLExecuteKernel_(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
				size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
				int depth, const char *build_options)
		{
			//construct kernel name
			//The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
			//for exmaple split_C2_D2, represent the split kernel with channels =2 and dataType Depth = 2(Data type is char)
			stringstream idxStr;
			if(channels != -1)
				idxStr << "_C" << channels;
			if(depth != -1)
				idxStr << "_D" << depth;
			kernelName += idxStr.str();

			cl_kernel kernel;
			kernel = openCLGetKernelFromSource(clCxt, source, kernelName, build_options);

			if ( localThreads != NULL)
			{
				globalThreads[0] = divUp(globalThreads[0], localThreads[0]) * localThreads[0];
				globalThreads[1] = divUp(globalThreads[1], localThreads[1]) * localThreads[1];
				globalThreads[2] = divUp(globalThreads[2], localThreads[2]) * localThreads[2];

				//size_t blockSize = localThreads[0] * localThreads[1] * localThreads[2];
				anakin::saber::openCLVerifyKernel(clCxt, kernel, localThreads);
			}
			for (size_t i = 0; i < args.size(); i++)
				// openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));
				clSetKernelArg(kernel, i, args[i].first, args[i].second);


#ifndef PRINT_KERNEL_RUN_TIME
			//openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
			//                                      localThreads, 0, NULL, NULL));
			clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
					localThreads, 0, NULL, NULL);
#else
			cl_event event = NULL;
			/* openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
			   localThreads, 0, NULL, &event));*/
			clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
					localThreads, 0, NULL, &event);

			cl_ulong start_time, end_time, queue_time;
			double execute_time = 0;
			double total_time   = 0;

			//openCLSafeCall(clWaitForEvents(1, &event));
			//openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
			//                                       sizeof(cl_ulong), &start_time, 0));

			//openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
			//                                       sizeof(cl_ulong), &end_time, 0));

			//openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
			//                                       sizeof(cl_ulong), &queue_time, 0));

			clWaitForEvents(1, &event);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
					sizeof(cl_ulong), &start_time, 0);

			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
					sizeof(cl_ulong), &end_time, 0);

			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
					sizeof(cl_ulong), &queue_time, 0);


			execute_time = (double)(end_time - start_time) / (1000 * 1000);
			total_time = (double)(end_time - queue_time) / (1000 * 1000);

			//	cout << setiosflags(ios::left) << setw(15) << execute_time;
			//	cout << setiosflags(ios::left) << setw(15) << total_time - execute_time;
			//	cout << setiosflags(ios::left) << setw(15) << total_time << endl;

			total_execute_time += execute_time;
			total_kernel_time += total_time;
			clReleaseEvent(event);
#endif

			clFinish(clCxt->impl->clCmdQueue);
			//openCLSafeCall(clReleaseKernel(kernel));
			clReleaseKernel(kernel);
		}
		void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName,
				size_t globalThreads[3], size_t localThreads[3],
				vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options)

		{
#ifndef PRINT_KERNEL_RUN_TIME
			openCLExecuteKernel_(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth,
					build_options);
#else
			string data_type[] = { "uchar", "char", "ushort", "short", "int", "float", "double"};
			cout << endl;
			cout << "Function Name: " << kernelName;
			if(depth >= 0)
				cout << " |data type: " << data_type[depth];
			cout << " |channels: " << channels;
			cout << " |Time Unit: " << "ms" << endl;

			total_execute_time = 0;
			total_kernel_time = 0;
			cout << "-------------------------------------" << endl;

			cout << setiosflags(ios::left) << setw(15) << "excute time";
			cout << setiosflags(ios::left) << setw(15) << "lauch time";
			cout << setiosflags(ios::left) << setw(15) << "kernel time" << endl;
			int i = 0;
			for(i = 0; i < RUN_TIMES; i++)
				openCLExecuteKernel_(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth,
						build_options);

			cout << "average kernel excute time: " << total_execute_time / RUN_TIMES << endl; // "ms" << endl;
			cout << "average kernel total time:  " << total_kernel_time / RUN_TIMES << endl; // "ms" << endl;
#endif
		}
		void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName,
				size_t globalThreads[3], size_t localThreads[3],
				vector< pair<size_t, const void *> > &args, int channels, int depth)
		{
			openCLExecuteKernel(clCxt, source, kernelName, globalThreads, localThreads, args,
					channels, depth, NULL);
		}


		double openCLExecuteKernelInterop(Context *clCxt , const char **source, string kernelName,
				size_t globalThreads[3], size_t localThreads[3],
				vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options, 
				bool finish, bool measureKernelTime, bool cleanUp)

		{
			//construct kernel name
			//The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
			//for exmaple split_C2_D2, represent the split kernel with channels =2 and dataType Depth = 2(Data type is char)
			stringstream idxStr;
			if(channels != -1)
				idxStr << "_C" << channels;
			if(depth != -1)
				idxStr << "_D" << depth;
			kernelName += idxStr.str();

			cl_kernel kernel;
			kernel = openCLGetKernelFromSource(clCxt, source, kernelName, build_options);

			double kernelTime = 0.0;

			if( globalThreads != NULL)
			{
				if ( localThreads != NULL)
				{
					globalThreads[0] = divUp(globalThreads[0], localThreads[0]) * localThreads[0];
					globalThreads[1] = divUp(globalThreads[1], localThreads[1]) * localThreads[1];
					globalThreads[2] = divUp(globalThreads[2], localThreads[2]) * localThreads[2];

					//size_t blockSize = localThreads[0] * localThreads[1] * localThreads[2];
					anakin::saber::openCLVerifyKernel(clCxt, kernel, localThreads);
				}
				for(size_t i = 0; i < args.size(); i ++)
					//openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));
					clSetKernelArg(kernel, i, args[i].first, args[i].second);

				if(measureKernelTime == false)
				{
					/*openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
					  localThreads, 0, NULL, NULL));*/
					clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
							localThreads, 0, NULL, NULL);
				}
				else
				{
					cl_event event = NULL;
					/* openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
					   localThreads, 0, NULL, &event));*/
					clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
							localThreads, 0, NULL, &event);

					cl_ulong end_time, queue_time;

					//openCLSafeCall(clWaitForEvents(1, &event));
					clWaitForEvents(1, &event);

					/*openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
					  sizeof(cl_ulong), &end_time, 0));*/
					clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
							sizeof(cl_ulong), &end_time, 0);

					kernelTime = (double)(end_time - queue_time) / (1000 * 1000);

					clReleaseEvent(event);
				}
			}

			if(finish)
			{
				clFinish(clCxt->impl->clCmdQueue);
			}

			if(cleanUp)
			{
				// openCLSafeCall(clReleaseKernel(kernel));
				clReleaseKernel(kernel);
			}

			return kernelTime;
		}

		// Converts the contents of a file into a string
		//static int convertToString(const char *filename, std::string& s)
		int convertToString(const char *filename, std::string& s)
		{
			size_t size;
			char*  str;

			std::fstream f(filename, (std::fstream::in | std::fstream::binary));
			if(f.is_open())
			{
				size_t fileSize;
				f.seekg(0, std::fstream::end);
				size = fileSize = (size_t)f.tellg();
				f.seekg(0, std::fstream::beg);

				str = new char[size+1];
				if(!str)
				{
					f.close();
					return -1;
				}

				f.read(str, fileSize);
				f.close();
				str[size] = '\0';

				s = str;
				delete[] str;
				return 0;
			}
			printf("Error: Failed to open file %s\n", filename);
			return -1;
		}

		double openCLExecuteKernelInterop(Context *clCxt , const char **fileName, const int numFiles, string kernelName,
				size_t globalThreads[3], size_t localThreads[3],
				vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options, 
				bool finish, bool measureKernelTime, bool cleanUp)

		{
			std::vector<std::string> fsource;
			for (int i = 0 ; i < numFiles ; i++)
			{
				std::string str;
				if (convertToString(fileName[i], str) >= 0)
					fsource.push_back(str);
			}
			const char **source = new const char *[numFiles];
			for (int i = 0 ; i < numFiles ; i++)
				source[i] = fsource[i].c_str();
			double kernelTime = openCLExecuteKernelInterop(clCxt ,source, kernelName, globalThreads, localThreads,
					args, channels, depth, build_options, finish, measureKernelTime, cleanUp);
			fsource.clear();
			delete []source;
			return kernelTime;
		}

		cl_mem load_constant(cl_context context, cl_command_queue command_queue, const void *value,
				const size_t size)
		{
			int status;
			cl_mem con_struct;

			con_struct = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
			// openCLSafeCall(status);

			clEnqueueWriteBuffer(command_queue, con_struct, 1, 0, size,
					value, 0, 0, 0) ;
			/*openCLSafeCall(clEnqueueWriteBuffer(command_queue, con_struct, 1, 0, size,
			  value, 0, 0, 0));*/

			return con_struct;

		}

		/////////////////////////////OpenCL initialization/////////////////
		//auto_ptr<Context> Context::clCxt;
		//int Context::val = 0;
		//std::mutex cs;
		std::mutex val_mutex;
		Context *Context::getContext()
		{
			//std::lock_guard<std::mutex> lock(val_mutex);
			//static Context con;
			//return &con;

			if(val == 0)
			{
				// AutoLock al(val_mutex);

				std::lock_guard<std::mutex> lock(val_mutex);

				if( NULL == clCxt.get())
					clCxt.reset(new Context);

				val = 1;
				return clCxt.get();
			}
			else
			{
				return clCxt.get();
			}
		}
		/*
		   Context *Context::getContext()
		   {
		//std::lock_guard<std::mutex> lock(val_mutex);
		//static Context con;
		//return &con;

		if (val == 0)
		{
		AutoLock al(val_mutex);

		//std::lock_guard<std::mutex> lock(val_mutex);




		if (NULL == clCxt.get())
		clCxt.reset(new Context);

		val = 1;
		return clCxt.get();
		}
		else
		{
		return clCxt.get();
		}
		}
		*/
		void Context::setContext(Info &oclinfo)
		{
			Context *clcxt = getContext();
			clcxt->impl->clContext = oclinfo.impl->oclcontext;
			clcxt->impl->clCmdQueue = oclinfo.impl->clCmdQueue;
			clcxt->impl->devices = oclinfo.impl->devices[oclinfo.impl->devnum];
			clcxt->impl->devName = oclinfo.impl->devName[oclinfo.impl->devnum];
			clcxt->impl->maxDimensions = oclinfo.impl->maxDimensions;
			clcxt->impl->maxWorkGroupSize = oclinfo.impl->maxWorkGroupSize;
			for(size_t i=0; i<clcxt->impl->maxDimensions && i<4; i++)
				clcxt->impl->maxWorkItemSizes[i] = oclinfo.impl->maxWorkItemSizes[i];
			clcxt->impl->maxComputeUnits = oclinfo.impl->maxComputeUnits;
			clcxt->impl->double_support = oclinfo.impl->double_support;
			//extra options to recognize compiler options
			memcpy(clcxt->impl->extra_options, oclinfo.impl->extra_options, 512);
			cl_bool unfymem = false;
			/*  openCLSafeCall(clGetDeviceInfo(clcxt->impl->devices, CL_DEVICE_HOST_UNIFIED_MEMORY,
			    sizeof(cl_bool), (void *)&unfymem, NULL));*/
			clGetDeviceInfo(clcxt->impl->devices, CL_DEVICE_HOST_UNIFIED_MEMORY,
					sizeof(cl_bool), (void *)&unfymem, NULL);
			if(unfymem)
				clcxt->impl->unified_memory = 1;
		}
		Context::Context()
		{
			impl = new Impl;
			//Information of the OpenCL context
			impl->clContext = NULL;
			impl->clCmdQueue = NULL;
			impl->devices = NULL;
			impl->maxDimensions = 0;
			impl->maxWorkGroupSize = 0;
			for(int i=0; i<4; i++)
				impl->maxWorkItemSizes[i] = 0;
			impl->maxComputeUnits = 0;
			impl->double_support = 0;
			//extra options to recognize vendor specific fp64 extensions
			memset(impl->extra_options, 0, 512);
			impl->unified_memory = 0; 
			programCache = ProgramCache::getProgramCache();
		}

		Context::~Context()
		{
			delete impl;
			programCache->releaseProgram();
		}
		Info::Info()
		{
			impl = new Impl;
			impl->oclplatform = 0;
			impl->oclcontext = 0;
			impl->clCmdQueue = 0;
			impl->devnum = 0;
			impl->maxDimensions = 0;
			impl->maxWorkGroupSize = 0;
			impl->maxWorkItemSizes = 0;
			impl->maxComputeUnits = 0;
			impl->double_support = 0;
			//extra_options = 0;
		}
		void Info::release()
		{
			//fft_teardown();
			if(impl->oclplatform)
			{
				impl->oclplatform = 0;
			}
			if(impl->clCmdQueue)
			{
				//openCLSafeCall(clReleaseCommandQueue(impl->clCmdQueue));
				clReleaseCommandQueue(impl->clCmdQueue);
			}
			ProgramCache::getProgramCache()->releaseProgram();
			if(impl->oclcontext)
			{
				//openCLSafeCall(clReleaseContext(impl->oclcontext));
				clReleaseContext(impl->oclcontext);

			}
			if(impl->maxWorkItemSizes)
			{
				delete[] impl->maxWorkItemSizes;
				impl->maxWorkItemSizes = 0;
			}
			//if(extra_options)
			//{
			//	delete[] extra_options;
			//	extra_options = 0;
			//}
			impl->devices.clear();
			impl->devName.clear();
			DeviceName.clear();
		}
		Info::~Info()
		{
			release();
			delete impl;
		}
		Info &Info::operator = (const Info &m)
		{
			impl->oclplatform = m.impl->oclplatform;
			impl->oclcontext = m.impl->oclcontext;
			impl->clCmdQueue = m.impl->clCmdQueue;
			impl->devnum = m.impl->devnum;
			impl->maxDimensions = m.impl->maxDimensions;
			impl->maxWorkGroupSize = m.impl->maxWorkGroupSize;
			impl->maxWorkItemSizes = m.impl->maxWorkItemSizes;
			impl->maxComputeUnits = m.impl->maxComputeUnits;
			impl->double_support = m.impl->double_support;
			memcpy(impl->extra_options, m.impl->extra_options, 512);
			for(size_t i = 0; i < m.impl->devices.size(); i++)
			{
				impl->devices.push_back(m.impl->devices[i]);
				impl->devName.push_back(m.impl->devName[i]);
				DeviceName.push_back(m.DeviceName[i]);
			}
			return *this;
		}
		Info::Info(const Info &m)
		{
			impl = new Impl;
			*this = m;
		}
	}//namespace anakin 

}//namespace saber 
