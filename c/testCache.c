
/*http:://igoro
 
#include <stdio.h>
#include <stdlib.h>
#include <linux/types.h>
#include <string.h>
#define SIZE_1KB (1024)
#define SIZE_1MB (1024*1024)
#define NUMBER 62*SIZE_1MB
#define MILLION 1000000
__u64 rdtsc()
{
	__u32 hi;
	__u32 lo;
	__asm__volatile__
	[M P>(
	“rdtsc”:"a"(lo),"="d"(hi)
	);



}



