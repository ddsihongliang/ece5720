#ifndef KERNEL_V2_H
#define KERNEL_V2_H
__global__ void calculate_forces(float4 *devX, float4 *devA, int n, int numTiles);

#endif
