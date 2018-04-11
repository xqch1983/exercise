#include<stdafx.h>
#include<iostream>
using namespace std;
class Node {
public:
	int data;
	Node *next;
	Node(int _data)
	{
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
	void Delete(int position);
	void Delete(Node _data);
	void Modify(int position, int _data);
	int Size();

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

void Linklist::Modify(int position, int _data)
{
	Node * current_node = head;

	if (position < 0 && position > Size())
	{

		cout << "the position is out the range of Linklist" << endl;
		return;

	}
	if (head == NULL && position == 0)
	{
		cout << "The linklist is Null" << endl;
		return;
	}
	for (int i = 1; i < position; i++)
	{
		current_node = current_node->next;
	}
	current_node->data = _data;

	cout << "the position and data to modify is " << position << _data << endl;

}


int Linklist::Size()
{
	int count = 0;
	if (head == NULL)
		return count;
	else
	{
		Node *current_head = head;
		while (current_head != NULL)
		{
			current_head = current_head->next;
			count++;
		}
		return count;
	}
}

void Linklist::Output()
{
	if (head == NULL)
		return;
	Node *current_node = head;
	while (current_node != NULL)
	{
		cout << current_node->data << endl;
		current_node = current_node->next;

	}
	cout << endl;
}


void Linklist::Delete(int position)
{
	if (head == NULL && position == 0)
		return;
	if (position > Size())
	{
		cout << "the position is big than linklist size" << endl;
		return;
	}

	Node *current_node = head;
	if (position == 1)
		head = current_node->next->next;

	else
	{
		for (int i = 0; i < position - 2; i++)
		{
			current_node = current_node->next;
		}
		cout << "deletet number is " << current_node->next->data << endl;
		current_node->next = current_node->next->next;
	}
}

void Linklist::Delete(Node node)
{
	if (head == NULL)
		return;
	Node *pre_current_node = head, *current_node = head;;
	while (current_node != NULL)
	{
		if (current_node->data != node.data)

		{
			pre_current_node = current_node;
			current_node = current_node->next;
		}
		else
		{
			cout << "deletet number is " << current_node->data << endl;
			pre_current_node->next = current_node->next;
			current_node = pre_current_node;

		}

	}
}
int main()
{
	Linklist linklist;
	for (int i = 0; i <= 10; i++)
	{
		Node *node = new Node(i + 1);
		linklist.Insert(node, i);
	}
	int size = linklist.Size();
	cout << "<<<the size of the linklist is>>>:" << size << endl;
	linklist.Output();
	Node oneNode(5);
	cout << "after delete xxxxxxxxxxxxxxxxxxxx" << endl;

	//linklist.Delete(5);
	linklist.Delete(oneNode);
	size = linklist.Size();
	cout << "After delete one item,the size of the linklist is:" << size << endl;
	linklist.Output();

	linklist.Modify(1, 10);
	cout << "after Modify xxxxxxxxxxxxxxxxxxxx" << endl;
	linklist.Output();
	return 0;


}
