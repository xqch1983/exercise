GCC=g++


CXXFLAGS :=-std=c++11 -g        ## don't use hip pinned host memory

PROGRAMS := anakinCache


HEAD_PATH = -I/opt/rocm/opencl/include/
LIB_PATH = -L/opt/rocm/opencl/lib/x86_64/
LIBS = -lOpenCL
all: $(PROGRAMS)

#Step

anakinCache: main.o initialization.o
	$(GCC)  $(CXXFLAGS)  $^ -o $@ $(LIB_PATH) $(LIBS)
	rm -f *.o

main.o: main.cpp    
	$(GCC) $(CXXFLAGS)  -c $< -o $@ $(HEAD_PATH)

initialization.o: initialization.cpp   
	$(GCC) $(CXXFLAGS) -c $< -o $@ $(HEAD_PATH)

clean:
	rm -f *.o $(PROGRAMS)


