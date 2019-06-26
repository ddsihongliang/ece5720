#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <cuda_runtime.h>

#define EPS2 1.0E-9

__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;

  // r_ij [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

  // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);

  // s = m_j * invDistCube [1 FLOP]
  float s = bj.w * invDistCube;

  //a_i= a_i+s*r_ij[6FLOPS]
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  //printf("ai.x : %f\n", ai.x);
  return ai;
}

__global__ void
calculate_forces(float4 *devX, float4 *devA, int N, int numTiles)
{
  extern __shared__ float4 shPosition[];

  float4 *globalX = devX;
  float4 *globalA = devA;
  float4 myPosition;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gtid >= N) return;
  myPosition = globalX[gtid];

  for (int i = 0; i < numTiles; i++) {
    int idx = i * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    //printf("myPosition.x = %f, shPosition[%d].x = %f, globalX[%d].x = %f\n", myPosition.x, threadIdx.x, shPosition[threadIdx.x].x, idx, globalX[idx].x);
    for (int j = 0; j < blockDim.x; j++) {
      acc = bodyBodyInteraction(myPosition, shPosition[j], acc);
    }
    __syncthreads();
  }

  // Save the result in global memory for the integration step.
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;
}
