//inverseFlipFloatInstantiated
//flipFloatInstantiated
#define cl_int    int
#define cl_uint   unsigned int
#define cl_short  short
#define cl_ushort unsigned short
#define cl_long   long
#define cl_ulong  unsigned long
#define cl_float  float
#define cl_double double
#define cl_char   char
#define cl_uchar  unsigned char

// Type Definitions

#define WG_SIZE 256 
#define ELEMENTS_PER_WORK_ITEM 4 
#define RADICES 16 
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE) 
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE) 
#define m_n x 
#define m_nWGs y 
#define m_startBit z 
#define m_nBlocksPerWG w 

#define CHECK_BOUNDARY 





__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void
flipFloatInstantiated(__global uint * restrict data, int4 cb_data)
{
	__local uint lmem[WG_SIZE];
	uint lIdx = get_local_id(0);
	uint wgIdx = get_group_id(0);

	const int n = cb_data.m_n;
	uint numBlocks = cb_data.m_nBlocksPerWG * ELEMENTS_PER_WORK_ITEM;

	int offset = (WG_SIZE*numBlocks)*wgIdx;
	//if (lIdx == 1 && wgIdx == 1)
	//	printf("numBlocks=%d,offset = %d\n", numBlocks, offset);
 #pragma unroll
    for (int i = 0; i<numBlocks; i++)
	{
		int addr = offset + i*WG_SIZE;
		uint value = ((addr + lIdx) < n) ? data[addr + lIdx] : 0;
		//if (wgIdx == 1 && lIdx == 1)
		//	printf("before lvalue = %d,addr=%d\n", value, addr);
		//unsigned int mask = (-(int)(value >> 31))  | 0x80000000;
		unsigned int mask = -(int)(value >> 31) | 0x80000000;
		//unsigned int mask = 0;// -(int(value >> 31)) | 0x80000000;
		value ^= mask;
		//if (wgIdx == 1 && lIdx == 1)
		//	printf("before lvalue = %d,addr=%d\n", value, addr);
		if ((addr + lIdx) < n)
			data[addr + lIdx] = value;
	}
}


__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void
inverseFlipFloatInstantiated(__global uint * restrict data, int4 cb_data)
{
	__local uint lmem[WG_SIZE];
	uint lIdx = get_local_id(0);
	uint gid = get_global_id(0);
	uint wgIdx = get_group_id(0);

	const int n = cb_data.m_n;
	uint numBlocks = cb_data.m_nBlocksPerWG * ELEMENTS_PER_WORK_ITEM;
	int offset = (WG_SIZE*numBlocks)*wgIdx;

	for (int i = 0; i<numBlocks; i++)
	{
		int addr = offset + i*WG_SIZE;
		uint value = ((addr + lIdx) < n) ? data[addr + lIdx] : 0;
		unsigned int mask = ((value >> 31) - 1) | 0x80000000;
		value ^= mask;
		if ((addr + lIdx) < n){
			data[addr + lIdx] = value;
		}
	}
}

