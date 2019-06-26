#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_types.h>
#include "body.h"
#include "render.h"
#include "kernel_v1.h"
#include "kernel_v2.h"

#define BLOCK_SIZE 1024
#define N 65535
#define ITERS 100
//#define DT 1000.0

int main(int argc, char const *argv[]) {
	
	int type = 0;
	cudaEvent_t start, stop;
	float elapsed_time = 0.0;
	float total_time = 0.0;
	
	if ( argc < 2 ){
		type = 0;
	} else if ( argc > 2 ) {
		printf("\n Too many arguments. \n");
		return -1;
	} else {
		type = atoi( argv[1] );
	}
  
	if (type == 1) {
		Body* bodies = (Body*) malloc(sizeof(Body));
		initialize(bodies, N);
		int bytes = N * sizeof(float);
		float* mass_buf;
		cudaMalloc(&mass_buf, bytes);

		float* pos_buf;
		cudaMalloc(&pos_buf, 3 * bytes);

		float* v_buf;
		cudaMalloc(&v_buf, 3 * bytes);

		int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
		create_frame(bodies->positions, bodies->velocity, N, 0);
		
		cudaEventCreate(&start);
  		cudaEventCreate(&stop);

		for (int i = 1; i <= ITERS; i++) {

			cudaEventRecord(start);

			cudaMemcpy(mass_buf, bodies->mass, bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(pos_buf, bodies->positions, (3 * bytes), cudaMemcpyHostToDevice);
			cudaMemcpy(v_buf, bodies->velocity, (3 * bytes), cudaMemcpyHostToDevice);

			bodyForce <<<nBlocks, BLOCK_SIZE >>> (pos_buf, v_buf, mass_buf, N);

			cudaMemcpy(bodies->mass, mass_buf, bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies->positions, pos_buf, (3 * bytes), cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies->velocity, v_buf, (3 * bytes), cudaMemcpyDeviceToHost);
			
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
  			cudaEventElapsedTime(&elapsed_time, start, stop);
			total_time += elapsed_time;

			for(int j = 0; j < 3 * N; j++) {
				bodies->positions[j] += bodies->velocity[j] * DT * 0.5;
			}	
			create_frame(bodies->positions, bodies->velocity, N, i);
		}
		printf("The CUDA execution time for %d iterations is %f msec.\n", ITERS, total_time);

	} else {
		Body* bodies = (Body*) malloc(sizeof(Body));
		initialize(bodies, N);
		float4 *positions = (float4*) malloc(sizeof(float4) * N);
		float4 *accelerations = (float4*) malloc(sizeof(float4) * N);
		float4 *g_pos;
		cudaMalloc(&g_pos, sizeof(float4) * N);
		float4 *g_acc;
		cudaMalloc(&g_acc, sizeof(float4) * N);
		int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
		
		cudaEventCreate(&start);
  		cudaEventCreate(&stop);
		
		for (int i = 1; i <= ITERS; i++) {
			
			cudaEventRecord(start);
			
			for (int j = 0; j < N; j++) {
				positions[j].x = bodies->positions[3 * j];
				positions[j].y = bodies->positions[3 * j + 1];
				positions[j].z = bodies->positions[3 * j + 2];
				positions[j].w = bodies->mass[j];
				accelerations[j].x = 0.0f;
				accelerations[j].y = 0.0f;
				accelerations[j].z = 0.0f;
				accelerations[j].w = 0.0f;
			}
			cudaMemcpy(g_pos, positions, sizeof(float4) * N, cudaMemcpyHostToDevice);
			cudaMemcpy(g_acc, accelerations, sizeof(float4) * N, cudaMemcpyHostToDevice);
			calculate_forces <<< nBlocks, BLOCK_SIZE, sizeof(float4) * BLOCK_SIZE >>> (g_pos, g_acc, N, nBlocks);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) 
    				printf("Error: %s\n", cudaGetErrorString(err));

			cudaMemcpy(positions, g_pos, sizeof(float4) * N, cudaMemcpyDeviceToHost);
			cudaMemcpy(accelerations, g_acc, sizeof(float4) * N, cudaMemcpyDeviceToHost);

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
  			cudaEventElapsedTime(&elapsed_time, start, stop);
			total_time += elapsed_time;

			for(int j = 0; j < N; j++) {
				bodies->velocity[3 * j] += accelerations[j].x * DT;
				bodies->velocity[3 * j + 1] += accelerations[j].y * DT;
				bodies->velocity[3 * j + 2] += accelerations[j].z * DT;
			}
			
			for (int k = 0; k < 3 * N; k++) {
				bodies->positions[k] += bodies->velocity[k] * DT * 0.5;
			}
			
			create_frame(bodies->positions, bodies->velocity, N, i);
		}
		printf("The CUDA execution time for %d iterations is %f msec.\n", ITERS, total_time);
	}

	return 0;
}
