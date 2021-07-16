#include<stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


#define NUM_ROWS 10000
#define NUM_COLS 1000
int	ha[NUM_ROWS][NUM_COLS] ;
int	hb[NUM_ROWS][NUM_COLS] ;
int	hc[NUM_ROWS][NUM_COLS] ;

 __global__ void add(int* da, int* db, int* dc){
    int tid = blockDim.x * blockDim.y * (blockIdx.y * gridDim.x + blockIdx.x) + (threadIdx.y * blockDim.x + threadIdx.x);
	while(tid<NUM_ROWS* NUM_COLS){
        dc[tid] = da[tid]+ db[tid];
        tid= tid + blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    }
}

int main(){
    int	*da;
    int *db;
    int *dc;
    int iSize = NUM_ROWS * NUM_COLS * sizeof(int) ;
    cudaError_t     cuError = cudaSuccess;
	int total=NUM_ROWS*NUM_COLS/512;
    dim3 dimGrid (total+1, 1, 1) ;
    dim3 dimBlock (512, 1, 1) ;

    for(int i=0;i<NUM_ROWS;i++){
        for(int j=0;j<NUM_COLS;j++){
            ha[i][j]=rand()%10+1;
	        hb[i][j]=rand()%10+1;
        }
    }

    cuError = cudaMalloc((void**)&da, iSize) ;
    if (cudaSuccess != cuError){
        printf ("Failed to allocate memory\n") ;
        return 1 ;
    }
    cuError = cudaMemcpy(da, ha, iSize, cudaMemcpyHostToDevice);
    if (cudaSuccess != cuError){
        cudaFree (da) ;
        printf ("Failed in Memcpy 1\n") ;
        return 1 ;
    }
	
    cuError = cudaMalloc((void**)&db, iSize) ;
    if (cudaSuccess != cuError){
        printf ("Failed to allocate memory\n") ;
        return 1 ;
    }
    cuError = cudaMemcpy(db, hb, iSize, cudaMemcpyHostToDevice);
    if (cudaSuccess != cuError){
        cudaFree (db) ;
        printf ("Failed in Memcpy 1\n") ;
        return 1 ;
    }
    cuError = cudaMalloc((void**)&dc, iSize) ;
    if (cudaSuccess != cuError){
        printf ("Failed to allocate memory\n") ;
        return 1 ;
    }


	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
    add<<<dimGrid, dimBlock>>>(da, db, dc);
	
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
	
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
    cuError = cudaGetLastError () ;
    if (cudaSuccess != cuError){
        printf ("Failed in kernel launch and reason is %s\n", cudaGetErrorString(cuError)) ;
        return 1 ;
    }

    cuError = cudaMemcpy(hc, dc, iSize, cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuError){
        cudaFree (dc) ;
        printf ("Failed in Memcpy 2\n") ;
        return 1 ;
    }

    bool success = true;
    for(int i=0;i<NUM_ROWS;i++){
        for(int j=0;j<NUM_COLS;j++){
            if ((ha[i][j] + hb[i][j]) != hc[i][j]){
		        printf( "Error:  %d + %d != %d\n", ha[i][j], hb[i][j], hc[i][j] );
		        success = false;
	        } 
	    }
    }
    if (success) printf( "We did it!\n" );
	
    cudaFree (da) ;
    cudaFree (db) ;	
    cudaFree (dc) ;

    return 0;
}