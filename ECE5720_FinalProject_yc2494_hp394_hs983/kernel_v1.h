#ifndef KERNEL_V1_H
#define KERNEL_V1_H

#define DT 1.0
__global__
void bodyForce(float *position, float *velocity, const float *mass, int n);

#endif
