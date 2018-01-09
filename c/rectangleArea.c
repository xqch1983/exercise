/*
阿里云，搜索引擎中5亿个url怎么高效存储；
一道C++笔试题，求矩形交集的面积：
在一个平面坐标系上，有两个矩形，它们的边分别平行于X和Y轴。
其中，矩形A已知， ax1(左边), ax2（右边）, ay1（top的纵坐标）, ay2（bottom纵坐标）. 矩形B，类似，就是 bx1, bx2, by1, by2。这些值都是整数就OK了。
要求是，如果矩形没有交集，返回-1， 有交集，返回交集的面积。
int area(rect const& a, rect const& b)
{
...
}
点评：
healer_kx：
补齐代码，最好是简洁的，别用库。你可以写你的辅助函数，宏定义，代码风格也很重要。
*/
 
struct rectangle
{
	double x[2];
	double y[2];

};
template <typename T> 
T const& min(T const&x, T const &y)
{
	return x < y ? x : y;
}
T const& max(T const&x, T const &y)
{
	return x < y ? y : x;
}

double area(rectangle const &a, rectangle const &b)
{
	double   dx = min(a.x[1], b.x[1]) - max(a.x[0], b.x[0]);
	double   dy = min(a.y[1], b.y[1]) - max(a.y[0], b.y[0]);
	return dx >= 0 && dy >= 0 ? dx*dy : -1;

}
