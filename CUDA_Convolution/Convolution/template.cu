#include <wb.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + (MASK_WIDTH - 1))
#define clamp(x) (min(max((x), 0.0), 1.0))

//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions in when loading input list elements into the shared memory

__global__ void convolution_2D_kernel(float *P, float *N, int imageHeight, int imageWidth, int channels, const float * __restrict__ M) {

	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tempY = by * O_TILE_WIDTH + ty;
	int tempX = bx * O_TILE_WIDTH + tx;

	for (int k = 0; k < channels; k++) {

		float accum = 0;
		int offset = ty * O_TILE_WIDTH + tx;
		int yOffset = offset / BLOCK_WIDTH;
		int xOffset = offset % BLOCK_WIDTH;
		int yIndex = by * O_TILE_WIDTH + yOffset - (MASK_WIDTH / 2);
		int xIndex = bx * O_TILE_WIDTH + xOffset - (MASK_WIDTH / 2);
		int index = (yIndex * imageWidth + xIndex) * channels + k;
		
		if (yIndex >= 0 && yIndex < imageHeight && 
			xIndex >= 0 && xIndex < imageWidth) {
			N_ds[yOffset][xOffset] = N[index];
		} else {
			N_ds[yOffset][xOffset] = 0.0f;
		}
				
		offset = ty * O_TILE_WIDTH + tx + (O_TILE_WIDTH * O_TILE_WIDTH);  
		yOffset = offset / BLOCK_WIDTH;
		xOffset = offset % BLOCK_WIDTH;
		yIndex = by * O_TILE_WIDTH + yOffset - (MASK_WIDTH / 2);
		xIndex = bx * O_TILE_WIDTH + xOffset - (MASK_WIDTH / 2);
		index = (yIndex * imageWidth + xIndex) * channels + k;
		
		if (yOffset < BLOCK_WIDTH && xOffset < BLOCK_WIDTH) {
			
			if (xIndex >= 0 && xIndex < imageWidth && 
				yIndex >= 0 && yIndex < imageHeight) {
				N_ds[yOffset][xOffset] = N[index];
			} else {
				N_ds[yOffset][xOffset] = 0.0f;
			}
			
		} else {}
		
		__syncthreads();
		
		for (int i = 0; i < MASK_WIDTH; i++) {
		
			for (int j = 0; j < MASK_WIDTH; j++) {
				accum += N_ds[ty + i][tx + j] * M[i * MASK_WIDTH + j];
			}
		
		}
		
		if (tempY < imageHeight && tempX < imageWidth) {
			P[(tempY * imageWidth + tempX) * channels + k] = clamp(accum);
		} else {}
		
		__syncthreads();
		
	}
}


int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //allocate device memory
  
  
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
  
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //copy host memory to device
  
  
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
  
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //initialize thread block and kernel grid dimensions
  //invoke CUDA kernel	
  
  
  dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);
  
  dim3 dimGrid(((imageWidth - 1) / O_TILE_WIDTH) + 1, ((imageHeight - 1) / O_TILE_WIDTH) + 1, 1);
  
  convolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);
  
  
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //copy results from device to host	
  
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //deallocate device memory	
  
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);
  
  
  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
