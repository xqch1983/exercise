//histogramAscInstantiated  yes
//scanInstantiated   yes

//histogramDescInstantiated
//histogramSignedAscInstantiated
//histogramSignedDescInstantiated
 

#define datatype float

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


uint scanlMemPrivData(uint val, __local uint* lmem, int exclusive)
{
	int lIdx = get_local_id(0);
	int wgSize = get_local_size(0);
	lmem[lIdx] = 0;

	lIdx += wgSize;
	lmem[lIdx] = val;
	barrier(CLK_LOCAL_MEM_FENCE);

	uint t;
	for (int i = 1; i < wgSize; i *= 2)
	{
		t = lmem[lIdx - i];
		barrier(CLK_LOCAL_MEM_FENCE);
		lmem[lIdx] += t;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return lmem[lIdx - exclusive];
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scanInstantiated(__global uint * isums,
	const int n,
	__local uint * lmem)
{
	__local int s_seed;
	s_seed = 0; barrier(CLK_LOCAL_MEM_FENCE);

	int last_thread = (get_local_id(0) < n &&
		(get_local_id(0) + 1) == n) ? 1 : 0;
	for (int d = 0; d < 16; d++)
	{
		uint val = 0;

		if (get_local_id(0) < n)
		{
			val = isums[(n * d) + get_local_id(0)];
		}
		uint res = scanlMemPrivData(val, lmem, 1);
		if (get_local_id(0) < n)
		{
			isums[(n * d) + get_local_id(0)] = res + s_seed;
		}

		if (last_thread)
		{
			s_seed += res + val;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void
histogramAscInstantiated(__global const uint * in,
	__global uint * isums,
	int4 cb)
{

	__local uint lmem[WG_SIZE*RADICES];

	uint gIdx = get_global_id(0);
	uint lIdx = get_local_id(0);
	uint wgIdx = get_group_id(0);
	uint wgSize = get_local_size(0);

	const int shift = cb.m_startBit;
	const int dataAlignment = 1024;
	const int n = cb.m_n;
	const int w_n = n + dataAlignment - (n%dataAlignment);

	const int nWGs = cb.m_nWGs;
	const int nBlocksPerWG = cb.m_nBlocksPerWG;

	for (int i = 0; i<RADICES; i++)
	{
		lmem[i*get_local_size(0) + lIdx] = 0;
	}
	GROUP_LDS_BARRIER;

	const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

	int nBlocks = (w_n) / blockSize - nBlocksPerWG*wgIdx;
	int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
	for (int iblock = 0; iblock<(((nBlocksPerWG) < (nBlocks)) ? (nBlocksPerWG) : (nBlocks)); iblock++)
	{
		for (int i = 0; i<ELEMENTS_PER_WORK_ITEM; i++, addr += WG_SIZE)
		{
#if defined(CHECK_BOUNDARY) 
			if ((addr) < n)
#endif 
			{
				uint local_key = (in[addr] >> shift) & 0xFU;
#if defined(DESCENDING) 
				lmem[(RADICES - local_key - 1)*get_local_size(0) + lIdx]++;
#else 
				lmem[local_key*get_local_size(0) + lIdx]++;
#endif 
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
#if 1
     
    if (lIdx < RADICES)
    {
        uint sum = 0;
        for (int i = 0; i<get_local_size(0); i++)        
        {
            sum += lmem[lIdx*get_local_size(0) + i];
        }
        isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;
    }
#else 
        uint sum = 0;
        //WG_SIZE == get_local_size(0)
        //Original RADICES=16 Trehads loop 256 times.
        //New: step1: every 16 threads sum 1 RADICE. 
        //            16 RADICES needs 256 threads
        //            Every Thread needs 16 LDS_read + 1 LDS_WRITE 
        //New: Step2:  reduce every 16 threads into 1 result 
        
        
        uint radice_idx =  (lIdx >> 4);
        uint offset =  (lIdx & 0xF);              
        for (int i = 0; i<WG_SIZE/RADICES; i++)        
        {                
            sum += lmem[radice_idx*WG_SIZE + i*16 + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lIdx] = sum;
        
        //Reduce: 16x to 4x  
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lIdx < 64){
            sum  = 0;
            radice_idx = (lIdx >> 2) ;
            offset     = (lIdx & 0x3); 
            sum  +=  lmem[radice_idx * 16 + 0 + offset];
            sum  +=  lmem[radice_idx * 16 + 4 + offset];
            sum  +=  lmem[radice_idx * 16 + 8 + offset];
            sum  +=  lmem[radice_idx * 16 + 12+ offset];
            lmem[lIdx] = sum;
        }
        //Reduce 4x to 1x
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lIdx < RADICES){
            sum  = 0;
            radice_idx = lIdx  ;
            offset     = 0; 
            sum  +=  lmem[radice_idx * 4 + 0];
            sum  +=  lmem[radice_idx * 4 + 1];
            sum  +=  lmem[radice_idx * 4 + 2];
            sum  +=  lmem[radice_idx * 4 + 3];        
          
          isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;    
        }
	
#endif
}



