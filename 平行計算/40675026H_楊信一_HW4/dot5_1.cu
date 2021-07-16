#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N   33*1024*1024 
#define THREADSPERBLOCK 1024
#define BLOCKSPERGRID (N+THREADSPERBLOCK-1)/THREADSPERBLOCK
__global__ void dot( int *a, int *b, int *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	__shared__ int cache[THREADSPERBLOCK];
	int cacheIndex = threadIdx.x;
    //c[tid] = a[tid] * b[tid];
	cache[cacheIndex] = a[tid] * b[tid];
	// synchronize threads in this block
    __syncthreads();
	i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];   
		}
		__syncthreads();
        i /= 2;
    }
	if(cacheIndex==0) c[blockIdx.x] = cache[0];
}
int main( void ) {
    int *a, *b, *partial_c;
    int *dev_a, *dev_b, *partial_dev_c;
	int dotCPU = 0;
	int dotGPU = 0;
    // allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
	partial_c = (int*)malloc( BLOCKSPERGRID * sizeof(int) );
    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    cudaMalloc( (void**)&partial_dev_c, BLOCKSPERGRID * sizeof(int) );
    // fill the arrays 'a' and 'b' on the CPU
	//srand ( time(NULL) );
    for (int i=0; i<N; i++) {
        a[i] = rand()%256;
        b[i] = rand()%256;
    }
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );
	// Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);	
    dot<<<BLOCKSPERGRID, THREADSPERBLOCK>>>( dev_a, dev_b, partial_dev_c);
	// Get stop time event    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    // copy the array 'partial_dev_c' back from the GPU to the CPU
    cudaMemcpy( partial_c, partial_dev_c, BLOCKSPERGRID * sizeof(int), cudaMemcpyDeviceToHost );
	for(int i = 0; i<BLOCKSPERGRID; i++){
		dotGPU += partial_c[i];
	}
    // verify that the GPU did the work we requested
    bool success = true;
	struct timespec t_start, t_end;
	double elapsedTimeCPU;
	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);	
    for (int i=0; i<N; i++) {
		dotCPU += a[i] * b[i];
    }
	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);
	// compute and print the elapsed time in millisec
	elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
	printf("Speedup: %.2lf\n", elapsedTimeCPU / elapsedTime );
	if (dotCPU != dotGPU) {
        printf( "Error: dotCPU %d != dotGPU %d\n", dotCPU, dotGPU );
        success = false;
    }
    if (success)    printf( "Test pass!\n" );
    // free the memory allocated on the GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
	cudaFree( partial_dev_c );
    //cudaFree( dev_dot );
    return 0;
}