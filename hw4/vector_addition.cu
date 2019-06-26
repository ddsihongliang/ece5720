// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void add( float *d_a, float *d_b, float *d_c, int N_num ) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  printf("threadID:%d\n",tid);
  if (tid < N_num) d_c[tid] = d_a[tid] + d_b[tid];

}

int main() {
  int N = 300;
  float h_a[N], h_b[N], h_c[N];
  float *dev_a, *dev_b, *dev_c;

  cudaMalloc( (void**)&dev_a, N * sizeof(float) );
  cudaMalloc( (void**)&dev_b, N * sizeof(float) );
  cudaMalloc( (void**)&dev_c, N * sizeof(float) );

  for (int i=0; i<N; i++) {
    dev_a[i] = ((float)rand())/RAND_MAX;
    dev_b[i] = ((float)rand())/RAND_MAX;
  }

  cudaMemcpy(dev_a,h_a,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,h_b,N*sizeof(float),cudaMemcpyHostToDevice);

  add<<<N,1>>>( dev_a, dev_b, dev_c, N);

  cudaMemcpy(h_c,dev_c,N*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree( dev_a ); cudaFree( dev_b ); cudaFree( dev_c );

  printf("Done\n");

  return 0;
}
