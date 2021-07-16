#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#define N 1000

int a[N][N], b[N][N], a1[N][N], b1[N][N], a2[N][N], b2[N][N], c1[N][N], c2[N][N], c3[N][N];
struct v {
   int i; /* row */
   int j; /* column */
};

void *runner(void *param); 

int main(){
	struct timespec t_start, t_end;
	double time1,time2,time3;
	int i, j, k,temp;
    pthread_t tid[N];       //Thread ID
	pthread_attr_t attr[N]; //Set of thread attributes
   
	for( i=0; i<N; i++ ){
		for( j=0; j<N; j++ ) {
			a[i][j] = rand()%10;
			b[i][j] = rand()%10;
			a1[i][j] = a[i][j];
			a2[i][j] = a[i][j];
			b1[i][j] = b[i][j];
			b2[i][j] = b[i][j];
			c1[i][j] = 0;
			c2[i][j] = 0;
			c3[i][j] = 0;
		}
	}
	
    // 1.standard
	clock_gettime( CLOCK_REALTIME, &t_start); 	
	for( i=0; i<N; i++ ){
        for( j=0; j<N; j++ ){
            for( k=0; k<N; k++ ){
                c1[i][j] += a[i][k]*b[k][j];
            }
        }
    }
	clock_gettime( CLOCK_REALTIME, &t_end);
	time1 = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	time1 += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("1. Sequential time: %lf ms\n", time1);
	
    // 2.pthread parallel+ transpose
	clock_gettime( CLOCK_REALTIME, &t_start);
	for( i=0; i<N; i++ ){
		for( j=i+1; j<N; j++ ) {
			temp = b1[j][i];
			b1[j][i] = b1[i][j];
			b1[i][j] = temp;
		}
	}
	for( i=0; i<100; i++ ) {
		struct v *data = (struct v *) malloc(sizeof(struct v));
		data->i = i * (N/100);
		pthread_attr_init(&attr[i]);
		pthread_create(&tid[i],&attr[i],runner,data);
	}
	for( i=0; i<100; i++ ) {
		pthread_join(tid[i], NULL);
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	time2 = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	time2 += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("2. pthread parrelel + transpose time: %lf ms\n", time2);
	printf("speedup:%lf\n",time1/time2);
	
    // 3.openmp parallel + transpose
	clock_gettime( CLOCK_REALTIME, &t_start);
	for( i=0; i<N; i++ ){
		for( j=i+1; j<N; j++ ) {
			temp = b2[j][i];
			b2[j][i] = b2[i][j];
			b2[i][j] = temp;
		}
	}
	#pragma omp parallel for private(j, k)
	for( i=0; i<N; i++ ){
		for( j=0; j<N; j++ ){
			int sum = 0;
			for( k=0; k<N; k++ ){
				sum += a2[i][k]*b2[j][k];
			}
			c3[i][j] = sum;
		}
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	time3 = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	time3 += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("3. OpenMp parrelel + transpose time: %lf ms\n", time3);
	printf("speedup:%lf\n",time1/time3);
	
	//check
    int pass = 1;
	for( i=0; i<N; i++ ){
        for( j=0; j<N; j++ ){
            if(c1[i][j] != c2[i][j]||c1[i][j] != c3[i][j]){
                pass = 0;
            }
        }
	}	
	if(pass == 1)
		printf("Test pass!\n");
	else
		printf("Test fail!\n");
	printf("\n");

   return 0;
}

void *runner(void *param) {
	struct v *data = param; 
	int k, sum = 0;
	for( int i=0; i<N/100 ;i++ ){
		for( int j=0; j<N; j++ ){ //整個row
			sum = 0;
			for( k=0; k<N; k++ ){ //個
				sum += a1[data->i][k] * b1[j][k];
			}
			c2[data->i][j] = sum;
		}
		data->i++;
	}
	pthread_exit(0);
}
