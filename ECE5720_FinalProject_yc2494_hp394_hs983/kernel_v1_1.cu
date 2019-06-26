#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DT 0.01
#define SOFTENING 1e-9f
#define BLOCK_SIZE 1024

__global__
void bodyForce(float *position, float *velocity, const float *mass, int n) {
  int i = 3*(blockDim.x * blockIdx.x + threadIdx.x);
  int j;
  float3 d,a;
  float distSqr,invDist,invDist3;

  if(i < n) {
    a.x = 0.0f; a.y = 0.0f; a.z = 0.0f;//initialize acceleration.
    for ( j = 0; j < 3*n; j=j+3) { // Step of 3.
      // calculate distance.
      d.x = position[j]   - position[i];
      d.y = position[j+1] - position[i+1];
      d.z = position[j+2] - position[i+2];
      distSqr = d.x*d.x + d.y*d.y + d.z*d.z + SOFTENING;
      invDist = 1.0f/sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      // calculate accel = mj*Rij*(invDist3).
      ax += d.x * invDist3 * mass[j/3];
      ay += d.y * invDist3 * mass[j/3];
      az += d.z * invDist3 * mass[j/3];
    }
    // Update velocity
    velocity[i]   += DT*ax;
    velocity[i+1] += DT*ay;
    velocity[i+2] += DT*az;
  }
  cudaDeviceSynchronize();
  for (j = 0; j < 3*n; j=j+3){
    //update position
    position[j]   += velocity[j]  *DT;
    position[j+1] += velocity[j+1]*DT;
    position[j+2] += velocity[j+2]*DT;
  }
}
