#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {

  //@@finding matrix dimension and matching with thread and matrix element
  int bx = blockIdx.x;	int by = blockIdx.y;
  int tx = threadIdx.x;	int ty = threadIdx.y;

  int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float Pvalue = 0;

  //@@declaring shared memory
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

  for (int p = 0; p < ((numAColumns - 1) / TILE_WIDTH + 1); ++p) {

    if (Row < numARows && p * TILE_WIDTH + tx < numAColumns) {
      ds_M[ty][tx] = A[Row * numAColumns + p * TILE_WIDTH + tx];
	} else {
	  ds_M[ty][tx] = 0.0;
	}
	
	if (p * TILE_WIDTH + ty < numBRows && Col < numBColumns){
      ds_N[ty][tx] = B[(p * TILE_WIDTH + ty) * numBColumns + Col];
	} else {
      ds_N[ty][tx] = 0.0;
	}

    __syncthreads(); //@@to keep track the shared memory as if all threads can be done at same time
	if (Row < numCRows && Col < numCColumns){
		for (int i = 0; i < TILE_WIDTH; ++i) {
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		}
		__syncthreads();
	}
  }

  if (Row < numCRows && Col < numCColumns) {
	  C[Row * numCColumns + Col] = Pvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C
  int numCColumns; // number of columns in the matrix C
  
  int sizeA, sizeB, sizeC;

  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;

  sizeA = numARows * numAColumns * sizeof(float);
  sizeB = numBRows * numBColumns * sizeof(float);
  sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  
  hostC = (float*) malloc(sizeC);
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory

  wbCheck(cudaMalloc((void**) &deviceA, sizeA));
  wbCheck(cudaMalloc((void**) &deviceB, sizeB));
  wbCheck(cudaMalloc((void**) &deviceC, sizeC));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU
  
  wbCheck(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions
  
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(numBColumns / TILE_WIDTH, numARows / TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel

  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU
  
  wbCheck(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory
  
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
