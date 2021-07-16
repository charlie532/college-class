#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main(){

      fork();
      printf("thread1\n");
      fork();
      printf("thread2\n");
      fork();
      printf("thread3\n");

      return 0;

}