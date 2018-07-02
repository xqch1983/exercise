#pragma OPENCL EXTENSION cl_amd_printf:enable
#define datatype float
__kernel void vscopy(__global datatype * a,  __global datatype * c)
{
	int global_id = get_global_id(0);
	c[global_id] = 1.1*a[global_id];
	//global_id = global_id * 2;
	//c[global_id] = 1.1*a[global_id]+c[global_id];
	if(global_id==0)
	printf("the num is %d\n",c[global_id]);

}
__kernel void vsadd(__global datatype * a, __global datatype * c)
{
	int global_id = get_global_id(0);
	c[global_id] = 1.1*a[global_id];
	//global_id = global_id * 2;
	//c[global_id] = 1.1*a[global_id]+c[global_id];
	if (global_id == 0)
		printf("the num is %d\n", c[global_id]);

}
__kernel void vssub(__global datatype * a, __global datatype * c)
{
	int global_id = get_global_id(0);
	c[global_id] = 1.1*a[global_id];
	//global_id = global_id * 2;
	//c[global_id] = 1.1*a[global_id]+c[global_id];
	if (global_id == 0)
		printf("the num is %d\n", c[global_id]);

}