/*
 * Introduction.
 *
 * Implement the simple operation x := ax for a vector x of type double
 * and a constant 'a'.
 *
 * This part introduces the kernel.
 *
 * Part 1. write a kernel of prototype
 *         __global__ void mykernel(double a, double * x)
 *         which performs the relevant operation for one block.
 * Part 2. in the main part of the program, declare and initialise
 *         variables of type dim3 to hold the number of blocks, and
 *         the number of threads per block. Use one block and
 *         THREADS_PER_BLOCK in the first instance.
 * Part 3. Generalise the kernel to treat any number of blocks,
 *         and problem sizes which are not a whole number of blocks.
 *
 * Training material originally developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <cassert>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

/* Error checking routine and macro. */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* The number of integer elements in the array */
#define ARRAY_LENGTH 1000

/* Suggested kernel parameters */
#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 256

/* TODO: Kernel to scale input */
__global__ void myKernel(double a, double *x)
{
  // First get index
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Check we are in range
  if (i < ARRAY_LENGTH)
  {
    // Apply operation
    x[i] = a * x[i];
  }
}

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_LENGTH * sizeof(double);

  double a = 2.0;       /* constant a */
  double *h_x = NULL;   /* input array (host) */
  double *h_out = NULL; /* output array (host) */

  /* Check we have a GPU, and get device name from the hipDeviceProp_t
   * structure. This is for information. */

  int ndevice = 0;
  int deviceNum = -1;
  hipDeviceProp_t prop;

  HIP_ASSERT(hipGetDeviceCount(&ndevice));

  if (ndevice == 0) {
    std::cerr << "No GPU available!" << std::endl;
    std::exit(0);
  }

  HIP_ASSERT(hipGetDevice(&deviceNum));
  HIP_ASSERT(hipGetDeviceProperties(&prop, deviceNum));
  std::cout << "Device " << deviceNum << " name: " << prop.name << std::endl;
  std::cout << "Maximum number of threads per block: "
            << prop.maxThreadsPerBlock << std::endl;

  /* allocate memory on host; assign some initial values */

  h_x = new double[ARRAY_LENGTH];
  h_out = new double[ARRAY_LENGTH];
  assert(h_x);
  assert(h_out);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    h_x[i] = 1.0 * i;
    h_out[i] = 0;
  }

  /* TODO: allocate memory on device */
  double *d_x = NULL; // device data
  HIP_ASSERT( hipMalloc(&d_x, ARRAY_LENGTH * sizeof(double)) );

  /* TODO: copy input array from host to GPU */
  HIP_ASSERT( hipMemcpy(d_x, h_x, ARRAY_LENGTH * sizeof(double), hipMemcpyHostToDevice) );

  /* TODO: write kernel */
  unsigned int nBlocks = ARRAY_LENGTH / THREADS_PER_BLOCK;
  if (nBlocks * THREADS_PER_BLOCK < ARRAY_LENGTH)
  {
    nBlocks++;
  }
  dim3 gridDim = {nBlocks, 1, 1};
  dim3 blockDim = {THREADS_PER_BLOCK, 1, 1};
  myKernel<<<gridDim, blockDim>>>(a, d_x);
  HIP_ASSERT( hipPeekAtLastError() );   // Check for errors in kernel launch
  HIP_ASSERT( hipDeviceSynchronize() ); // Check for errors in the kernel

  /* TODO: copy the result array back to the host output array */
  HIP_ASSERT( hipMemcpy(h_out, d_x, ARRAY_LENGTH * sizeof(double), hipMemcpyDeviceToHost) );

  /// /* TODO: check what happens when device and host references are reversed */
  /// HIP_ASSERT( hipMemcpy(d_x, h_out, ARRAY_LENGTH * sizeof(double), hipMemcpyDeviceToHost) );

  /* We can now check the results ... */
  std::cout << "Results:" << std::endl;
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      // std::cout << std::setw(9) << i << " " << std::fixed
      //           << std::setprecision(2) << std::setw(5) << h_out[i]
      //           << std::endl;
      if (fabs(h_out[i] - a * h_x[i]) < DBL_EPSILON)
        ncorrect += 1;
    }
    std::cout << "No. elements " << ARRAY_LENGTH
              << ", and correct: " << ncorrect << std::endl;
  }

  /* TODO: free device buffer */
  HIP_ASSERT( hipFree(d_x) );

  /* free host buffers */

  delete h_x;
  delete h_out;

  return 0;
}

/* It is important to check the return code from API calls, so the
 * follow function/macro allow this to be done concisely as
 *
 *   HIP_ASSERT(hipRunTimeAPIFunction(...));
 *
 * Return codes may be asynchronous, and thus misleading! */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal) {

  if (ifail != hipSuccess) {
    std::cerr << "Line " << line << " (" << file
              << "): " << hipGetErrorName(ifail) << ": "
              << hipGetErrorString(ifail) << std::endl;
    if (fatal)
      std::exit(ifail);
  }

  return;
}
