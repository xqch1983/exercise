/*
* his file is copied from https://blog.csdn.net/l_tudou/article/details/51565271
* data: 2018-05-18 
* editor: 
*/

#include<stdio.h>  
#include<string.h>  
#include<unistd.h>  

void progress()  
{  
    char arr[103];  
    char *index="|/-\\0";  
    int i=0;  
    memset(arr,'\0',sizeof(arr));  
    arr[0]='[';  
    arr[101]=']';  
    arr[102]='\0';  
    for(i;i<=100;++i)  
    {   
        arr[i+1]='=';  
        usleep(20000);
        printf("%-101s[%d%%][%c]\r",arr,i,index[i%4]);  
        fflush(stdout);  
        usleep(100);  

    }   
    printf("\n"); 
}
int main()
{
    progress();
    return 0;

}
