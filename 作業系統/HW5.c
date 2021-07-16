// 40675026h_楊信一_HW5
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#define N 500

int a[N][N], b[N][N], bT[N][N], c1[N][N], c2[N][N], c3[N][N], c4[N][N];
struct v {
   int i; /* row */
   int j; /* column */
};

void *runner(void *param); 

int main(){
	struct timespec t_start, t_end;
	double elapsedTime;
	int i, j, k,temp;
    pthread_t tid[N];       //Thread ID
	pthread_attr_t attr[N]; //Set of thread attributes
   
	for( i=0; i<N; i++ ){
		for( j=0; j<N; j++ ) {
			a[i][j] = rand()%10;
			b[i][j] = rand()%10;
			bT[i][j] = b[i][j];
		}
	}
	
    // 1.standard
	clock_gettime( CLOCK_REALTIME, &t_start); 	
	for( i=0; i<N; i++ ){
        for( j=0; j<N; j++ ){
            c1[i][j] = 0;
            for( k=0; k<N; k++ ){
                c1[i][j] += a[i][k]*b[k][j];
            }
        }
    }
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("1. Sequential elapsedTime: %lf ms\n", elapsedTime);

    // 2.pthread parallel
	clock_gettime( CLOCK_REALTIME, &t_start);
	for( i=0; i<N; i++ ){
		for( j=0; j<N; j++ ) {
			c2[i][j] = 0;
		}
	}
	for( i=0; i<4; i++ ) {
		struct v *data = (struct v *) malloc(sizeof(struct v));
		data->i = i * (N/4);
		pthread_attr_init(&attr[i]);
		pthread_create(&tid[i],&attr[i],runner,data);
	}
	for( i=0; i<4; i++ ) {
		pthread_join(tid[i], NULL);
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("2. parrelel elapsedTime: %lf ms\n", elapsedTime);

    // 3.openmp parallel
	clock_gettime( CLOCK_REALTIME, &t_start);
	#pragma omp parallel shared(a,b,c) private(i,j,k)
	{
		#pragma omp for schedule(dynamic)
		{
			// #pragma omp parrelel for
			for( i=0; i<N; i++ ){
				for( j=0; j<N; j++ ){
					c3[i][j] = 0;
					for( k=0; k<N; k++ ){
						c3[i][j] += a[i][k]*b[k][j];
					}
				}
			}
		}
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("3. OMP elapsedTime: %lf ms\n", elapsedTime);

	// 4.transpose
	clock_gettime( CLOCK_REALTIME, &t_start);
	for( i=0; i<N; i++ ){
		for( j=i+1; j<N; j++ ) {
			temp = bT[j][i];
			bT[j][i] = b[i][j];
			bT[i][j] = temp;
		}
	}
	for( i=0; i<N; i++ ){
		for( j=0; j<N; j++ ) {
			c4[i][j] = 0;
			for( k=0; k<N; k++ ){
				c4[i][j] += a[i][k]*bT[j][k];
			}
		}
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("4. Transpose Sequential elapsedTime: %lf ms\n", elapsedTime);

    // check
    int pass = 1;
	for( i=0; i<N; i++ ){
        for( j=0; j<N; j++ ){
            if(c1[i][j] != c2[i][j]){
                pass = 0;
            }
        }
	}	
	if(pass == 1)
		printf("Test pass!\n");
	else
		printf("Test fail!\n");
	printf("N = %d\n",N);

	// for( i=0; i<N; i++ ){
	// 	printf("%d ",c[0][i]);
	// }
	// printf("\n");
	// for( i=0; i<N; i++ ){
	// 	printf("%d ",cc[0][i]);
	// }
	printf("\n");

   return 0;
}

void *runner(void *param) {
	struct v *data = param; 
	int k, sum = 0;
	for( int i=0; i<N/4 ;i++ ){
		for( int j=0; j<N; j++ ){ //整個row
			sum = 0;
			for( k=0; k<N; k++ ){ //個
				sum += a[data->i][k] * b[k][j];
			}
			c2[data->i][j] = sum;
		}
		data->i++;
	}
	pthread_exit(0);
}