#ifndef __CACHE_OCL_HPP__
#define __CACHE_OCL_HPP__
#include <map>
#include<memory>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <CL/opencl.h>
#include <string> 
typedef long long  int64;
namespace anakin
{
    namespace saber 
    {
        enum
        {
            OCL_DEVICE_TYPE_DEFAULT     = (1 << 0),
            OCL_DEVICE_TYPE_CPU         = (1 << 1),
            OCL_DEVICE_TYPE_GPU         = (1 << 2),
            OCL_DEVICE_TYPE_ACCELERATOR = (1 << 3),
            //OCL_DEVICE_TYPE_CUSTOM      = (1 << 4)
            OCL_DEVICE_TYPE_ALL         = 0xFFFFFFFF
        };

        enum DevMemRW
        {
            DEVICE_MEM_R_W = 0, 
            DEVICE_MEM_R_ONLY, 
            DEVICE_MEM_W_ONLY
        };
 
        enum DevMemType
        { 
            DEVICE_MEM_DEFAULT = 0, 
            DEVICE_MEM_AHP,         //alloc host pointer
            DEVICE_MEM_UHP,         //use host pointer
            DEVICE_MEM_CHP,         //copy host pointer
            DEVICE_MEM_PM           //persistent memory
        };
		enum soureFileType
		{
			SOURCE_FILE_TXT = 0,
			SOURCE_FILE_BINARY        //persistent memory
		};

        //Get the global device memory and read/write type	
        //return 1 if unified memory system supported, otherwise return 0
       
        //Set the global device memory and read/write type, 
        //the newly generated oclMat will all use this type
        //return -1 if the target type is unsupported, otherwise return 0
        
        //this class contains ocl runtime information
        class  Info
        {
        public:
            struct Impl;
            Impl *impl;

            Info();
            Info(const Info &m);
            ~Info();
            void release();
            Info &operator = (const Info &m);
            std::vector<std::string> DeviceName;
        };
        //////////////////////////////// Initialization & Info ////////////////////////
        //this function may be obsoleted
        //CV_EXPORTS cl_device_id getDevice();
        //the function must be called before any other cv::ocl::functions, it initialize ocl runtime
        //each Info relates to an OpenCL platform
        //there is one or more devices in each platform, each one has a separate name
        int getDevice(std::vector<Info> &oclinfo, int devicetype = OCL_DEVICE_TYPE_GPU);

        //set device you want to use, optional function after getDevice be called
        //the devnum is the index of the selected device in DeviceName vector of INfo
        void setDevice(Info &oclinfo, int devnum = 0);

        //optional function, if you want save opencl binary kernel to the file, set its path
        void setBinpath(const char *path);

        //The two functions below enable other opencl program to use ocl module's cl_context and cl_command_queue
        void* getoclContext();

        void* getoclCommandQueue();

        //this function enable ocl module to use customized cl_context and cl_command_queue
        //getDevice also need to be called before this function
        void setDeviceEx(Info &oclinfo, void *ctx, void *qu, int devnum = 0);

        //////////////////////////////// Error handling ////////////////////////
        void error(const char *error_string, const char *file, const int line, const char *func);

        //////////////////////////////// OpenCL context ////////////////////////
        //This is a global singleton class used to represent a OpenCL context.
        class Context
        {
        protected:
            Context();
            friend class std::auto_ptr<Context>;
            static std::auto_ptr<Context> clCxt;

        public:
            ~Context();
            static int val;
            static Context *getContext();
            static void setContext(Info &oclinfo);
	    struct Impl
	    {
		    //Information of the OpenCL context
		    cl_context clContext;
		    cl_command_queue clCmdQueue;
		    cl_device_id devices;
		    std::string devName;
		    cl_uint maxDimensions;
		    size_t maxWorkGroupSize;
		    size_t maxWorkItemSizes[4];
		    cl_uint maxComputeUnits;
		    int double_support;
		    //extra options to recognize vendor specific fp64 extensions
		    char extra_options[512];
		    std::string Binpath;
		    int unified_memory; //1 means integrated GPU, otherwise this value is 0
	    };
	    Impl *impl;
	};


/************************************************************************/

    	cl_kernel openCLGetKernelFromSource(const Context *clCxt, const char **source,std::string kernelName, const char *build_options);
	cl_kernel openCLGetKernelFromSource_2(const Context *clCxt, const char *sourceFile,std::string kernelName, const char *build_options);
	cl_kernel openCLGetKernelFromSource(const Context *clCxt, const char **source, std::string kernelName);

	static int convertToString(const char *filename, std::string& s);
	int savetofile(const Context *clcxt, cl_program &program, const char *fileName);

        class ProgramCache
        {
        protected:
            ProgramCache();
            friend class std::auto_ptr<ProgramCache>;
            static std::auto_ptr<ProgramCache> programCache;

        public:
            ~ProgramCache();
            static ProgramCache *getProgramCache()
            {
                if( NULL == programCache.get())
                    programCache.reset(new ProgramCache());
                return programCache.get();
            }

            //lookup the binary given the file name
            cl_program progLookup(std::string srcsign);

            //add program to the cache
            void addProgram(std::string srcsign, cl_program program);
            void releaseProgram();

            std::map <std::string, cl_program> codeCache;
            unsigned int cacheSize;
            //The presumed watermark for the cache volume (256MB). Is it enough?
            //We may need more delicate algorithms when necessary later.
            //Right now, let's just leave it along.
            static const unsigned MAX_PROG_CACHE_SIZE = 1024;
        };
	/**********************    class kernelCache        **********************************/
	class KernelCache
	{
	protected:
		KernelCache();
		friend class std::auto_ptr<KernelCache>;
		static std::auto_ptr<KernelCache> kernelCache;

	public:
		~KernelCache();
		static KernelCache *getKernelCache()
		{
			if (NULL == kernelCache.get())
			kernelCache.reset(new KernelCache());
			return kernelCache.get();
		}
		using Key = std::pair<std::string, std::string>;
		//lookup the binary given the file name
		cl_kernel kernelLookup(std::pair<std::string, std::string>  kernelName);

		//add kernel to the cache
		//void addKernel(string kernelName, cl_kernel kernel);
		void addKernel(std::pair<std::string, std::string>  kernelName, cl_kernel kernel);
		void releaseKernel();

		//using KernelMap = std::unordered_map<Key, cl_kernel, SimpleHash>;

		//map <string, cl_kernel> mapKernelCache;
		std::map <Key, cl_kernel> mapKernelCache;

		unsigned int cachKernelSize;
		//The presumed watermark for the cache volume (256MB). Is it enough?
		//We may need more delicate algorithms when necessary later.
		//Right now, let's just leave it along.
		static const unsigned MAX_KERNEL_CACHE_SIZE = 1024;
	};

    }//namespace saber
}//namespace anakin
#endif
