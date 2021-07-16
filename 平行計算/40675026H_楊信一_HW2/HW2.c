#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define N 1048576
//1048576
void serial_bitonic_convert(int start, int length, int *p, int flag);
void parallel_bitonic_convert(int start, int length, int *p, int flag);
void serial_bitonic_sort(int list_n, int n,  int *p);
void parallel_bitonic_sort(int list_n, int n, int *p);
void swap(int *a, int *b);
int test_sort(int *a, int *b);
int s[N], ss[N];

int main(){
	struct timespec t_start, t_end;
	double elapsedTime;
	int i, j, flag;
	srand(0);
	for(i = 0; i < N; i++){
		s[i] = rand()%N;
		ss[i] = s[i];
	}
	//serial
	clock_gettime( CLOCK_REALTIME, &t_start);
	for (i = 2; i <= N/2; i = 2 * i){
        for (j = 0; j < N; j += i){
            if ((j / i) % 2 == 0)
                flag = 1;
            else
                flag = 0;
            serial_bitonic_convert(j, i, s, flag);
        }
    }
	//printf("finish\n");
    serial_bitonic_sort(1, N, s);
	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("Sequential elapsedTime: %lf ms\n", elapsedTime);
	//parallel
	clock_gettime( CLOCK_REALTIME, &t_start);
	for (i = 2; i <= N/2; i = 2 * i){
		#pragma omp parallel for private(j, flag)
        for (j = 0; j < N; j += i){
            if ((j / i) % 2 == 0)
                flag = 1;
            else
                flag = 0;
            parallel_bitonic_convert(j, i, ss, flag);
        }
    }
    parallel_bitonic_sort(1, N, ss);
	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("Parallel elapsedTime: %lf ms\n", elapsedTime);
	
	if(test_sort(s, ss))
		printf("Test pass!\n");
	else
		printf("Test failed!\n");
	return 0;
}
void serial_bitonic_convert(int start, int length, int *p, int flag){
    int i;
    int div_len;
    if (length == 1)
        return;
    if (length % 2 !=0 ){
        printf("error\n");
        exit(0);
    }
    div_len = length / 2;
    for (i = start; i < start + div_len; i++){
        if (flag == 1){
            if (p[i] > p[i+div_len])
                swap(&p[i], &p[i+div_len]);
        }
        else{
            if (p[i] < p[i+div_len])
                swap(&p[i], &p[i+div_len]);
        }
    }
    serial_bitonic_convert(start, div_len, p, flag);
    serial_bitonic_convert(start + div_len, div_len, p, flag);
}
void serial_bitonic_sort(int list_n, int n, int *p){
	long long pos, i, j;
	while(list_n <= n){
		for(i = 0; i < list_n; i++){
			pos = i * (n / list_n);
			for(j = pos; j < pos + n / list_n / 2; j++){
				if(p[j] > p[n / list_n / 2 + j])
					swap(&p[j], &p[n / list_n / 2 + j]);
			}
		}
		list_n = list_n*2;
	}
}
void parallel_bitonic_convert(int start, int length, int *p, int flag)
{
    int i;
    int div_len;
    if (length == 1)
        return;
    if (length % 2 !=0 ){
        printf("error\n");
        exit(0);
    }
    div_len = length / 2;
	//#pragma omp parallel for shared(p, flag, start, div_len) private(i)
    for (i = start; i < start + div_len; i++){
        if (flag == 1){
            if (p[i] > p[i+div_len])
                swap(&p[i], &p[i+div_len]);
        }
        else{
            if (p[i] < p[i+div_len])
                swap(&p[i], &p[i+div_len]);
        }
    }
	if(div_len)
    serial_bitonic_convert(start, div_len, p, flag);
    serial_bitonic_convert(start + div_len, div_len, p, flag);
}
void parallel_bitonic_sort(int list_n, int n, int *p){
	long long pos, i, j;
	while(list_n <= n){
		for(i = 0; i < list_n; i++){
			pos = i * (n / list_n);
			for(j = pos; j < pos + n / list_n / 2; j++){
				if(p[j] > p[n / list_n / 2 + j])
					swap(&p[j], &p[n / list_n / 2 +j]);
			}
		}
		list_n = list_n*2;
	}
}
void swap(int *a, int *b){
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
int test_sort(int *a, int *b){
	int i;
	for(i = 0; i < N; i++){			
		if(a[i] != b[i])
			return 0;
	}	
	return 1;
}