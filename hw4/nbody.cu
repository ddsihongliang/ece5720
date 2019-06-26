/***********************************************************************
To Compile:
 /usr/local/cuda-10.0/bin/nvcc -arch=compute_52 -o file.out filename.cu
***********************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define TILE 32
void check_output(float *A, int dim) {
  /* Print output for debug */
  int i,j;
  printf("\n");
  for(i = 0; i < dim; i++) {
    for(j = 0; j < dim; j++) {
      printf("%3.4f ", A[i*dim + j]);
    }
    printf(";\n");
  }
  printf("\n");
}

__global__ void Navie_noShare(float *d_a,float *d_bT,float *d_c,int dim){
//  __shared__ float shared[1024];
  float partial = 0.0;

  int i = threadIdx.y + blockIdx.y * blockDim.y; //row i of c
  int j = threadIdx.x + blockIdx.x * blockDim.x; //Column j of c
  int k;
  i = i*dim;

  for(k = 0; k < dim; k++){
    partial+=d_a[i+k] * d_bT[j+k];
  }
  d_c[i+j] = partial;
}

__global__ void Tile_shared1(float *d_a,float *d_b,float *d_c,int dim){
  __shared__ float *a_tile[TILE*TILE];  // Static shared memory size.
  __shared__ float *b_tile[TILE*TILE];

  float partial = 0.0;
  int bx = blockIdx.x ; int by = blockIdx.y ;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int i = by * blockDim.y + ty; //row i of c
  int j = bx * blockDim.x + tx; //Column j of c
  int k,m;
  i  = i  * dim;
  y = ty*TILE;
  for (int m = 0; m < N/blockDim.x; ++m) {
    a_tile[y+tx] = d_a[i + m*blockDim.x + tx]; /* load coalesced */
    b_tile[y+tx] = d_b[(m*blockDim.y+ty)*dim+j]; /* not coalesced */
    __syncthreads();
    for(k = 0; k < blockDim.x; ++k)
      partial += a_tile[y+k] * b_tile[k*TILE+tx]; /* B bank conflicts */
    __syncthreads();
    d_c[i+j] = partial;
  }
}

int main(int argc, char const *argv[]) {
  // Initiailize matrix dimension
  int dim = 1024,block_size = 32;
  int i;
  if (argc > 1) {
    dim = atoi(argv[1]);
    block_size = atoi(argv[2]);
  }
  // declear host and device timer.
  srand(3);
  dim3 Block(TILE,TILE);
  dim3 Grid(dim/Block.x, dim/Block.y);
  struct timespec start,finish;
    int ntime, stime;
  float tot_time=0.0;

  // Populate matrice
  float *a  = (float*)malloc(sizeof(float)*dim*dim);
  float *bT = (float*)malloc(sizeof(float)*dim*dim);
  float *c  = (float*)malloc(sizeof(float)*dim*dim);
  float *d_a, *d_bT ,*d_c, limit=10.0; //d_bT for transposed

  for(i = 0; i < dim*dim; i++){
    a[i]  = ((float)rand()/(float)(RAND_MAX)) * limit;
    bT[i] = ((float)rand()/(float)(RAND_MAX)) * limit;
  }

  // Allocate device memeory.
  cudaMalloc( (void**)&d_a,  dim*dim*sizeof(float));
  cudaMalloc( (void**)&d_bT, dim*dim*sizeof(float));
  cudaMalloc( (void**)&d_c,  dim*dim*sizeof(float));

  // Initiailize timer & start recording.
  clock_gettime(CLOCK_REALTIME, &start);

  // Copy memory to device.
  cudaMemcpy(d_a ,a ,dim*dim*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_bT,bT,dim*dim*sizeof(float),cudaMemcpyHostToDevice);

  // Call CUDA kernel function.
  Navie_noShare<<<Grid, Block>>>(d_a,d_bT,d_c,dim);
  cudaMemcpy(c, d_c, sizeof(float)*dim*dim,cudaMemcpyDeviceToHost);

  // Timer stop.
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_REALTIME, &finish);
  ntime = finish.tv_nsec - start.tv_nsec;
  stime = (int)finish.tv_sec - (int) start.tv_sec;
  tot_time = ntime*1.0E-9 + stime;

  /* Print output for debug */
  printf("kernel#1 Time elapsed: %f ms. matrix dimension: %d X %d\n",
  tot_time*1.0E3,dim,dim);

  // reset memory and timer.
  cudaFree(d_c); cudaFree(d_bT); cudaFree(d_a);

  /*----------------Tile method with bank conflicts:------------------------*/
  // Allocate memory again:
  cudaMalloc( (void**)&d_a,  dim*dim*sizeof(float));
  cudaMalloc( (void**)&d_bT, dim*dim*sizeof(float));
  cudaMalloc( (void**)&d_c,  dim*dim*sizeof(float));

  // start timming.
  clock_gettime(CLOCK_REALTIME, &start);

  // Copy memory to device.
  cudaMemcpy(d_a ,a ,dim*dim*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_bT,bT,dim*dim*sizeof(float),cudaMemcpyHostToDevice);

  Tile_shared1<<<Grid, Block,2*Block.x*Block.y*sizeof(float)>>>(d_a,d_bT,d_c,dim);
  cudaMemcpy(c, d_c, sizeof(float)*dim*dim,cudaMemcpyDeviceToHost);

  // Timer stop.
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_REALTIME, &finish);
  ntime = finish.tv_nsec - start.tv_nsec;
  stime = (int)finish.tv_sec - (int) start.tv_sec;
  tot_time = ntime*1.0E-9 + stime;

  /* Print output for debug */
  printf("kernel#2 Time elapsed: %f ms. matrix dimension: %d X %d\n",
  tot_time*1.0E3,dim,dim);

  // check_output(c,dim);
  cudaFree(d_c); cudaFree(d_bT); cudaFree(d_a);
  free(a); free(bT); free(c);
  return 0;
}
