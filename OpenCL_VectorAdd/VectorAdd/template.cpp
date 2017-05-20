#include <wb.h>
#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//@@ OpenCL kernel
//const char *kernelSource = "";

const char *kernelSource =
"__kernel void vadd(	__global const float *a,	\n"
"						__global const float *b,	\n"
"						__global float *result,		\n"
"						unsigned int n)				\n"
"{													\n"
"	int id = get_global_id(0);						\n"
"	if (id < n)										\n"
"	{												\n"
"		result[id] = a[id] + b[id];					\n"
"	}												\n"
"}													\n";


int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;
  deviceInput1 = NULL;
  deviceInput2 = NULL;
  deviceOutput = NULL;

  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  context = NULL;        
  queue = NULL;    
  program = NULL;
  kernel = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  inputLengthBytes = inputLength * sizeof(float);
  hostOutput       = (float *)malloc(inputLengthBytes);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The input size is ", inputLengthBytes, " bytes");

  cl_int clerr = CL_SUCCESS;
  
  //@@ Initialize the workgroup dimensions

  size_t localSize = 64;
  size_t globalSize = ceil(inputLength / (float) localSize) * localSize;
  
  //@@ Bind to platform
  
  clerr = clGetPlatformIDs(1, &cpPlatform, NULL);

  //@@ Get ID for the device
  
  clerr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  
  //@@ Create a context
  
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &clerr);
  
  //@@ Create a command queue

  queue = clCreateCommandQueue(context, device_id, 0, &clerr);
  
  //@@ Create the compute program from the source buffer
  
  program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &clerr);
  
  //@@ Build the program executable

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  //@@ Create the compute kernel
  
  kernel = clCreateKernel(program, "vadd", &clerr);

  //@@ Create the input and output arrays in device memory for calculation
  
  deviceInput1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput1, NULL);
  deviceInput2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput2, NULL);
  deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputLengthBytes, NULL, NULL);

  //@@ Write data set into the input array in device memory
  
  clerr = clEnqueueWriteBuffer(queue, deviceInput1, CL_TRUE, 0, inputLengthBytes, hostInput1, 0, NULL, NULL);
  clerr |= clEnqueueWriteBuffer(queue, deviceInput2, CL_TRUE, 0, inputLengthBytes, hostInput2, 0, NULL, NULL);

  //@@ Set the arguments to compute kernel

  clerr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceInput1);
  clerr |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceInput2);
  clerr |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceOutput);
  clerr |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &inputLength);
  
  //@@ Execute the kernel over the entire range of the data set
  
  clerr = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

  //@@ Wait for the command queue to get serviced before reading back results
  
  clFinish(queue);

  //@@ Read the results from the device

  clEnqueueReadBuffer(queue, deviceOutput, CL_FALSE, 0, inputLengthBytes, hostOutput, 0, NULL, NULL);
  
  wbSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
