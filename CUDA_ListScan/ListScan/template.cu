#include <wb.h>

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	//@@ add scanned block sums to all values of the scanned blocks
	
	unsigned int tx = threadIdx.x;
	unsigned int b = (2 * blockIdx.x * BLOCK_SIZE);
	
	if (blockIdx.x > 0) {
		if (b + tx < len) {
			input[b + tx] += aux[blockIdx.x - 1];
		}
		if (b + BLOCK_SIZE + tx < len) {
			input[b + BLOCK_SIZE + tx] += aux[blockIdx.x - 1];
		}
	}
}
__global__ void scan(float *input, float *output, float *aux, int len) {
    //@@ generate the scanned blocks
    //@@ workefficient version of the parallel scan
    //@@ store the block sum to the aux array 
	
	//-------------------------------------------------------------------
	// Load a segment of the input vector into shared memory
	__shared__ float XY[BLOCK_SIZE * 2];
	
	unsigned int tx = threadIdx.x;
	unsigned int b = (2 * blockIdx.x * BLOCK_SIZE);
	
	if (b + tx < len) { // first half
		XY[threadIdx.x] = input[b + tx];
	} else {
		XY[threadIdx.x] = 0;
	}
	if (b + BLOCK_SIZE + tx < len) { // second half
		XY[BLOCK_SIZE + threadIdx.x] = input[b + BLOCK_SIZE + tx];
	} else {
		XY[BLOCK_SIZE + threadIdx.x] = 0;
	}
	
	__syncthreads();
	
	// Reduction Phase Kernel Code
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		//__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE) {
			XY[index] += XY[index - stride];
		}
		__syncthreads();
	}
	
	// Post Reduction Reverse Phase Kernel Code
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		//__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
		__syncthreads();
	}
	__syncthreads();
	if (b + tx < len) {
		output[b + tx] = XY[threadIdx.x];
	}

	if (b + BLOCK_SIZE + tx < len) {
		output[b + BLOCK_SIZE + tx] = XY[BLOCK_SIZE + threadIdx.x];
	}
	
	if (aux && threadIdx.x == 0) {
			aux[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
	}
	
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output 1D list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the input/output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate device memory
  
  cudaMalloc((void**) &deviceInput, numElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numElements * sizeof(float));
  
  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE * 2) * sizeof(float));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE * 2) * sizeof(float));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  //@@ Clear output memory
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy host memory to device  
  
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions
  
  int g = ceil((float) numElements / (BLOCK_SIZE * 2));
  dim3 dimGrid(g, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ launch scan kernel twice: for generating scanned blocks 
  //@@ (by passing deviceAuxArray to the aux parameter)
  //@@ and for generating scanned aux array that has the scanned block sums. 
  //@@ (by pass NULL to the aux parameter)
  //@@ Then call addScannedBlockSums kernel.
  
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxArray, numElements);
  cudaDeviceSynchronize();
  scan<<<dim3(1, 1, 1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, (BLOCK_SIZE * 2));
  cudaDeviceSynchronize();
  addScannedBlockSums<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy results from device to host	
  
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Deallocate device memory
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
