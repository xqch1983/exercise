 

#ifndef TEST_TEST_UTILS_HPP_
#define TEST_TEST_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>
#include "ff_timer.h"
 
#define RAND_MAX  99999
namespace test_utils
{ 
 void compareResults(double CPUtime, double GPUtime, int trial) {
	double time_ratio = (CPUtime / GPUtime);
	std::cout << "VERSION " << trial << " -----------" << std::endl;
	std::cout << "CPU time: " << CPUtime << std::endl;
	std::cout << "GPU time: " << GPUtime << std::endl;
	std::cout << "GPU is ";
	if (time_ratio > 1)
		std::cout << time_ratio << " times faster!" << std::endl;
	else
		std::cout << (1 / time_ratio) << " times slower :(" << std::endl;
}

 
  

// key ascending order
bool cmp1(pair<int, int>a, pair<int, int>b)
{
	return a.first < b.first;
}

int RandomNumber_int() { return (rand() % 9999); }
float RandomNumber_float() { return (rand() % 9999 + 1.12); }
  
// class generator:
struct c_unique {
	int current;
	c_unique() { current = 0; }
	int operator()() { return ++current; }
} UniqueNumber;



 
template<class T>
bool is_eq(const std::vector<T>& result, const std::vector<T>& expected)
{
	if (result.size() != expected.size())
		std::cout << "the size of two vector is different" << std::endl;
	 
    for(size_t i = 0,k=0; i < result.size(); k++,i++)
    {
		if (result[i] != expected[i])
		{
			k++;
			std::cout << "......GPU result\t" << result[i] << "\t CPU expected\t" << expected[i] << "\t for " << i << std::endl;
			 
		}
		if (k == 0)
			return true;
		else
			return false;
    }
}

  
template<class T,class T2>
void  prinprintOutput(const std::vector<T>& output, const std::vector<T2>& output2)
{
	if (output.size() != output2.size())
		std::cout << "the size of two vector is different" << std::endl;

	for (size_t i = 0, k = 0; i < output.size(); k++, i++)
	{
	 
			std::cout << "......key \t" << output[i] << "\t values\t" << output2[i] << "\t for index = " << i << std::endl;

	}
		 
} 

 
 



} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
