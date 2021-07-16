#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 1000000000
#define KERNELSIZE 9
#define THREADSPERBLOCK 1024
#define BLOCKSPERGRID (N+THREADSPERBLOCK-1)/THREADSPERBLOCK


// 9 wide 1d kernel, no padding so it cuts out early
// shift by 4 to align with original data
__global__ void conv( float *data, float *kernel, float *output ){
    int tid =blockIdx.x*blockDim.x+threadIdx.x;
	int i;

	for(i=0; i<9; i++){
		output[tid] += data[tid + i] * kernel[i];
	}
}

int main(){
	srand(time(NULL));
    struct timespec t_start, t_end;
	double elapsedTimeCPU;
	int pass = 1;

	// gassian kernel from: http://dev.theomader.com/gaussian-kernel-calculator/
	float kernel[9] = {0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229};


	// random number from python, irl this would come from lidar
	float* data = (float*)malloc(N*sizeof(float));
	for(int i = 0;i < N;i++){
		data[i] = rand()%10+rand()/RAND_MAX;
	}

	// empty array to store the output
	// float output[N-KERNELSIZE+1];

	//CPU
    clock_gettime( CLOCK_REALTIME, &t_start); 
	float* output = (float*)malloc((N-KERNELSIZE+1)*sizeof(float));
	for (int i = 0; i < N-KERNELSIZE+1;i++){
		output[i] = 0;
		for (int j = 0; j < KERNELSIZE; j++){
			output[i] += kernel[j] * data[i+j];
		}
	}
    clock_gettime( CLOCK_REALTIME, &t_end);
    elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);


	//GPU
	float *d_kernel, *d_data, *d_output;
	// allocate the memory on the GPU
    cudaMalloc( (void**)&d_kernel, KERNELSIZE * sizeof(float) );
    cudaMalloc( (void**)&d_data, N * sizeof(float) );
	cudaMalloc( (void**)&d_output, (N-KERNELSIZE+1) * sizeof(float) );
	float* output_from_device = (float*)malloc((N-KERNELSIZE+1)*sizeof(float));
	cudaMemcpy( d_kernel, kernel, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data, data, N * sizeof(float), cudaMemcpyHostToDevice );

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	conv<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_data, d_kernel, d_output);
  	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
	cudaMemcpy(output_from_device, d_output, (N-KERNELSIZE+1) * sizeof(float), cudaMemcpyDeviceToHost );

	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU elapsedTime: %lf ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("speedup: %lf \n", elapsedTimeCPU/elapsedTime);
	for (int i = 0; i < N-KERNELSIZE+1; i++){
		if(output_from_device[i]-output[i]>0.00001){ //don't use if(output_from_device[i]!=output[i])
			printf("CPU:%lf    GPU:%lf\n",output[i], output_from_device[i] );
			pass = 0;
		}
	}
	if(pass == 1)
		printf("Test pass!\n");
	else
		printf("Test fail!\n");
}