#include "iostream.h"
void main()
{
	cout<<"Hello World!"<<endl;
}




//example 2
#include <iostream.h>
#include <math.h>
class CalcSqrt
{
	public:
		CalcSqrt(float x)
		{
			y = x;
		
		
		}
		void print()
		{
			cout<<sqrt(y)<<endl;
			cout<<sqrt(y)<<endl;
		}
	private:
		float y;
};
void main()
{
	CalcSqrt x(9);
	x.print();
}
		
