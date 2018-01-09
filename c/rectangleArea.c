/*
�����ƣ�����������5�ڸ�url��ô��Ч�洢��
һ��C++�����⣬����ν����������
��һ��ƽ������ϵ�ϣ����������Σ����ǵı߷ֱ�ƽ����X��Y�ᡣ
���У�����A��֪�� ax1(���), ax2���ұߣ�, ay1��top�������꣩, ay2��bottom�����꣩. ����B�����ƣ����� bx1, bx2, by1, by2����Щֵ����������OK�ˡ�
Ҫ���ǣ��������û�н���������-1�� �н��������ؽ����������
int area(rect const& a, rect const& b)
{
...
}
������
healer_kx��
������룬����Ǽ��ģ����ÿ⡣�����д��ĸ����������궨�壬������Ҳ����Ҫ��
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
