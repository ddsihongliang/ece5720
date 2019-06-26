#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DT 0.01
#define SOFTENING 1e-9f

__global__
void bodyForce(float *position, float *velocity, const float *mass, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  if(i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for ( j = 0; j < n; j++) {
      float dx = position[j]   - position[i];
      float dy = position[j+1] - position[i+1];
      float dz = position[j+2] - position[i+2];
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
      // calculate force = mj*Rij*(invDist3).
      Fx += dx * invDist3 * mass[i];
      Fy += dy * invDist3 * mass[i];
      Fz += dz * invDist3 * mass[i];
    }
    // Update velocity
    velocity[i]   += DT*Fx;
    velocity[i+1] += DT*Fy;
    velocity[i+2] += DT*Fz;
  }
  for (j = 0; j < n; j++){
    //update position
    position[i]   += velocity[i]  *DT;
    positiop[i+1] += velocity[i+1]*DT;
    position[i+2] += velocity[i+2]*DT;
  }
}
