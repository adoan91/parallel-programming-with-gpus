#include <wb.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 512 

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
	unsigned int num_elements,
	unsigned int num_bins) {
	//@@ privitization technique
	
	__shared__ unsigned int histo_private[NUM_BINS];

	int i = threadIdx.x + blockIdx.x * blockDim.x; // global thread id
	// total number of threads
	int stride = blockDim.x * gridDim.x;

	if (threadIdx.x < num_bins) {
		histo_private[threadIdx.x] = 0;
	}

	__syncthreads();

	// compute block's histogram

	while (i < num_elements) {
		int temp = input[i];
		atomicAdd(&(histo_private[temp]), 1);
		i += stride;
	}
	// wait for all other threads in the block to finish
	__syncthreads();

	// store to global histogram

	if (threadIdx.x < num_bins) {
		//int t = histo_private[threadIdx.x];
		atomicAdd(&(bins[threadIdx.x]), histo_private[threadIdx.x]);
	}
	

	/*
	for (int pos = threadIdx.x; pos < NUM_BINS; pos += blockDim.x) {
		histo_private[pos] = 0;
	}
	__syncthreads();
	for (int pos = i; pos < num_elements; pos += stride) {
		atomicAdd(&(histo_private[input[i]]), 1);
	}
	__syncthreads();
	for (int pos = threadIdx; pos < NUM_BINS; pos += blockDim.x) {
		atomicAdd(&(bins[threadIdx.x]), histo_private[threadIdx.x]);
	}
	*/

	/*
	histo_private[threadIdx.x] = 0;
	__syncthreads();
	while (i < num_elements) {
		atomicAdd(&histo_private[input[i]], 1);
		i += stride;
	}
	__syncthreads();
	atomicAdd(&bins[threadIdx.x], histo_private[threadIdx.x]);
	*/
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ counters are saturated at 127
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i < num_bins) {
		if (bins[i] > 127) { // || bins[i] == 0
			bins[i] = 127;
		}
	}
	
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory

	//-------------------------------------------------------------------
	cudaMalloc((void**)&deviceInput, inputLength * sizeof(unsigned int));
	cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(unsigned int));
	//-------------------------------------------------------------------

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU

	//-------------------------------------------------------------------
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//-------------------------------------------------------------------

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(GPU, "Clearing the bins.");
	//@@ zero out the deviceBins

	//-------------------------------------------------------------------
	cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
	//-------------------------------------------------------------------

	wbTime_stop(GPU, "Clearing the bins.");

	//@@ Initialize the grid and block dimensions

	//-------------------------------------------------------------------
	// (NUM_BINS / BLOCK_SIZE)
	//ceil((float)inputLength / BLOCK_SIZE)
	dim3 dimGrid(ceil((float) (inputLength) / (BLOCK_SIZE)), 1, 1);
	dim3 dimBlock(BLOCK_SIZE*2, 1, 1);

	//-------------------------------------------------------------------

	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Perform kernel computations

	//-------------------------------------------------------------------
	histogram <<<dimGrid, dimBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
	saturate <<<dimGrid, dimBlock>>>(deviceBins, NUM_BINS);
	//-------------------------------------------------------------------

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU

	//-------------------------------------------------------------------
	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//-------------------------------------------------------------------

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory

	//-------------------------------------------------------------------
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	//-------------------------------------------------------------------

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
	return 0;
}
