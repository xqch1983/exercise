#include<stdafx.h>
#include<iostream>
using namespace std;
class Node {
public:
	int data;
	Node *next;
	Node(int _data) {
		data = _data;
		next = NULL;
	}
};
class Linklist {
public:
	Linklist() {
		head = NULL;

	}
	void Insert(Node *node, int position);
	void Output();
private:
	Node * head;
};
void Linklist::Insert(Node *node, int position)
{
	if (head == NULL) {
		head = node;
		return;
	}
	if (position == 0)
	{
		node->next = head;
		head = node;
		return;
	}
	Node * current_node = head;
	int i = 0;
	while (current_node->next != NULL && i < position - 1)
	{
		current_node = current_node->next;
		i++;
	}
	if (i == position - 1)
	{
		node->next = current_node->next;
		current_node->next = node;
	}
}
void Linklist::Output()
{
	if (head == NULL)
		return;
	Node *current_node = head;
	while (current_node != NULL)
	{
		cout << current_node->data;
		current_node = current_node->next;
	}
	cout << endl;
}
int main()
{
	Linklist linklist;
	for (int i = 0; i <= 10; i++)
	{
		Node *node = new Node(i);
		linklist.Insert(node, i - 1);
	}
	linklist.Output();
	return 0;
}
