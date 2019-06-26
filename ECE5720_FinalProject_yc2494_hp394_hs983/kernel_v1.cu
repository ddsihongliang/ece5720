#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel_v1.h"

//#define DT 100000.0
#define SOFTENING 1e-9f
#define BLOCK_SIZE 1024

__global__
void bodyForce(float *position, float *velocity, const float *mass, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  float3 d,a;
  float distSqr,invDist,invDist3;

  if(i < n) {
    a.x = 0.0f; a.y = 0.0f; a.z = 0.0f;//initialize acceleration.
    for ( j = 0; j < n; j++) { // Step of 3.
      if(i == j) continue;
      // calculate distance.
      d.x = position[j*3]   - position[i*3];
      d.y = position[j*3+1] - position[i*3+1];
      d.z = position[j*3+2] - position[i*3+2];
      distSqr = d.x*d.x + d.y*d.y + d.z*d.z + SOFTENING;
      invDist = 1.0f/sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      // calculate accel = mj*Rij*(invDist3).
      a.x += d.x * invDist3 * mass[j];
      a.y += d.y * invDist3 * mass[j];
      a.z += d.z * invDist3 * mass[j];
    }
//    printf("previous velocity %d: [%f, %f, %f]\n", i, position[i*3],position[i*3+1],position[i*3+2]);
    // Update velocity
//    printf("acc %d: [%f, %f, %f, %f]\n", i, a.x, a.y,a.z, mass[i]);
    
    velocity[i*3]   += DT*a.x;
    velocity[i*3+1] += DT*a.y;
    velocity[i*3+2] += DT*a.z;
  //  printf("after velocity %d: [%f, %f, %f]\n", i, velocity[i*3],velocity[i*3+1],velocity[i*3+2]);
  }
//  cudaDeviceSynchronize();
//  for (j = 0; j < 3*n; j=j+3){
    //update position
//    position[j]   += velocity[j]  *DT;
//    position[j+1] += velocity[j+1]*DT;
//    position[j+2] += velocity[j+2]*DT;
 // }
}
