#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define N 2000

int a[N][N], b[N][N], c[N][N], cc[N][N];

int min(int a, int b);

int main(){
	struct timespec t_start, t_end;
	double elapsedTime;
	int ii, jj, kk, i, j, k, temp;
	for( i=0; i<N; i++ )
		for( j=0; j<N; j++ ) {
			a[i][j] = rand();
			b[i][j] = rand();
			c[i][j] = 0;
			cc[i][j] = 0;
		}
	for( i=0; i<N; i++ ){
		for( j=i+1; j<N; j++ ) {
			temp = b[j][i];
			b[j][i] = b[i][j];
			b[i][j] = temp;
		}
	}
	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);  
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){
				c[i][j] += a[i][k]*b[j][k];
			}
		}
	}
	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);
	// compute and print the elapsed time in millisec
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("Sequential elapsedTime: %lf ms\n", elapsedTime);
	
	
	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);
	#pragma omp parallel for private(j, k)
	for( i = 0; i < N; ++i ){
		for( j = 0; j < N; ++j ) {
			for( k = 0; k < N; ++k ){
				cc[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);
	// compute and print the elapsed time in millisec
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("Parallel elapsedTime: %lf ms\n", elapsedTime);
	
	
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			if(cc[i][j] != c[i][j])
				break;
		}
	}
	if(i==N && j==N)
		printf("Test pass!!!\n"); 
	else
		printf("Test failed!!!\n");
	return 0;
}