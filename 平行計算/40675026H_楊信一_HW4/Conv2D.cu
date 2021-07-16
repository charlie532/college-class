#include <stdio.h>
#include <math.h>
#include <stdbool.h> 
#define WIDTH1 10000
#define WIDTH2 3
unsigned char input[WIDTH1*WIDTH1];
unsigned char output[WIDTH1*WIDTH1];
unsigned char outputGPU[WIDTH1*WIDTH1];//used to store results from GPU

bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY);

__global__ void convolve2D_GPU( unsigned char *input, unsigned char *output, double *kernel);
 

int main(){
	// unsigned char input[WIDTH1*WIDTH1] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	double Kernel[WIDTH2*WIDTH2] ={0,-1,0,-1,5,-1,0,-1,0};

	
	unsigned char *d_input;
	double *d_kernel;
	unsigned char *d_output;
	int i, j;

	srand(0);

	for (int i =0;i < WIDTH1*WIDTH1; i++)
	{
		input[i]= rand()%100;
		//printf("%d ",input[i]);
	}

	struct timespec t_start, t_end;
	double elapsedTimeCPU;
	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);
	
	convolve2DSlow(input, output, WIDTH1, WIDTH1, Kernel, WIDTH2, WIDTH2);
	
	//CPU end time
	clock_gettime( CLOCK_REALTIME, &t_end);
	// compute and print the elapsed time in millisec
	elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);

	// allocate the memory on the GPU
    cudaMalloc( (void**)&d_input, WIDTH1 * WIDTH1 * sizeof(unsigned char) );
    cudaMalloc( (void**)&d_kernel, WIDTH2 * WIDTH2 * sizeof(double) );
    cudaMalloc( (void**)&d_output, WIDTH1 * WIDTH1 * sizeof(unsigned char) );
	
	// copy the arrays 'input' and 'kernel' to the GPU
    cudaMemcpy( d_input, input, WIDTH1 * WIDTH1 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( d_kernel, Kernel, WIDTH2 * WIDTH2 * sizeof(double), cudaMemcpyHostToDevice );
	
    int total = WIDTH1*WIDTH1/512 +1 ;
	dim3 dimGrid (total, 1, 1) ;
    dim3 dimBlock (512, 1, 1) ;

	//GPU time
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	convolve2D_GPU<<<dimGrid, dimBlock>>>(d_input, d_output, d_kernel);
	//GPU end time
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	// copy the array 'd_output' back from the GPU to the CPU
    cudaMemcpy( outputGPU, d_output, WIDTH1 * WIDTH1 * sizeof(unsigned char), cudaMemcpyDeviceToHost );

	//check results
	int pass = 1;
	for(i=0; i<WIDTH1; i++){
		for(j=0; j<WIDTH1; j++){
			if(output[i*WIDTH1 + j] != outputGPU[i*WIDTH1 + j]){
				pass = 0;

			}
			// else printf("%8d ", output[i*WIDTH1 + j]);
		}
		// printf("\n");
	}
	
	if(pass==0) printf("Test Fail!\n");
	else printf("Test pass!\n");
	return 0;
}

bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY)
{
	int i, j, m, n, mm, nn;
	int kCenterX, kCenterY; // center index of kernel

	double sum; // temp accumulation buffer

	int rowIndex, colIndex;

	// check validity of params

	if(!in || !out || !kernel) return false;

	if(dataSizeX <= 0 || kernelSizeX <= 0) return false;

	// find center position of kernel (half of kernel size)

	kCenterX = kernelSizeX / 2;

	kCenterY = kernelSizeY / 2;

	for(i=0; i < dataSizeY; ++i){
		for(j=0; j < dataSizeX; ++j){
			sum = 0; // init to 0 before sum
			for(m=0; m < kernelSizeY; ++m){
				mm = kernelSizeY - 1 - m;
				for(n=0; n < kernelSizeX; ++n){
					nn = kernelSizeX - 1 - n; 

					// index of input signal, used for checking boundary
					
					rowIndex = i + (kCenterY - mm);
					colIndex = j + (kCenterX - nn);

					// ignore input samples which are out of bound

					if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
						sum += in[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * m + n];
					
				}

			}

			out[dataSizeX * i + j] = (unsigned char)(fabs(sum) + 0.5f);
		}		

	}

	return true;

}

__global__ void convolve2D_GPU( unsigned char *input, unsigned char *output, double *kernel){
 //    int i = (blockDim.x*blockIdx.x+threadIdx.x )/ WIDTH1;
	// int j = (blockDim.x*blockIdx.x+threadIdx.x ) % WIDTH1;
	int i,j;
	for(i = (blockDim.x*blockIdx.x+threadIdx.x )/ WIDTH1;i<WIDTH1;i+=blockDim.x*gridDim.x)
	{
		for(j=(blockDim.x*blockIdx.x+threadIdx.x)%WIDTH1;j<WIDTH1;j+=blockDim.x*gridDim.x)
		{
			int m, n, mm, nn;
			int kCenterX, kCenterY; // center index of kernel

			double sum; // temp accumulation buffer

			int rowIndex, colIndex;
			
			int dataSizeX, dataSizeY, kernelSizeX, kernelSizeY;
			dataSizeX = dataSizeY = WIDTH1;
			kernelSizeX = kernelSizeY = WIDTH2;
			// find center position of kernel (half of kernel size)
			kCenterX = kernelSizeX / 2;
			kCenterY = kernelSizeY / 2;
			
			sum = 0; // init to 0 before sum
			for(m=0; m < kernelSizeY; ++m){
				mm = kernelSizeY - 1 - m;
				for(n=0; n < kernelSizeX; ++n){
					nn = kernelSizeX - 1 - n; 

					// index of input signal, used for checking boundary
					rowIndex = i + (kCenterY - mm);
					colIndex = j + (kCenterX - nn);

					// ignore input samples which are out of bound

					if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
						sum += input[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * m + n];

				}

			}
			output[dataSizeX * i + j] = (unsigned char)(fabs(sum) + 0.5f);
			
			// j+=blockDim.x*gridDim.x;
		}
		// i+=blockDim.x*gridDim.x;
		// j = (blockDim.x*blockIdx.x+threadIdx.x ) % WIDTH1;
	}


}