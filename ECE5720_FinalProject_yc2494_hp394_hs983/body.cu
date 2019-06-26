#include "body.h"
#include <stdlib.h>
//#include <random>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define POSITION_MIN -500.0f
#define POSITION_MAX 500.0f

#define MIN_LEN 400.0f
#define MAX_LEN 600.0f
#define PI 3.1415926536f

#define VELOCITY_MIN -10.0f
#define VELOCITY_MAX 10.0f

#define MASS_MIN 0.1f
#define MASS_MAX 40.0f

void initialize(Body* b, int num) {
	b->size = num;
	b->mass = (float*)malloc(sizeof(float) * num);
	b->positions = (float*)malloc(sizeof(float) * num * 3);
	b->velocity = (float*)malloc(sizeof(float) * num * 3);

	srand(static_cast <unsigned> (time(0)));

	for (int i = 0; i < num; i++) {
		float r = (float)rand()/ (float)(RAND_MAX);
		b->mass[i] = MASS_MIN + r * (MASS_MAX - MASS_MIN);
	}
/*
	for (int i = 0; i < num * 3; i++) {
		float r1 = (float)(rand()) / (float)(RAND_MAX);
		b->positions[i] = POSITION_MIN + r1 * (POSITION_MAX - POSITION_MIN);
//		if(i % 3 == 0) printf("]\n[");
//		printf("%f, ", b->positions[i] + 500.0);
	}
*/	
	for (int i = 0; i < num; i++) {
		float r = (float)rand()/ (float)(RAND_MAX);
		float len = MIN_LEN + r * (MAX_LEN - MIN_LEN);
		r = (float)rand()/ (float)(RAND_MAX);
		float fy = r * PI;	
		r = (float)rand()/ (float)(RAND_MAX);
		float theta = r * PI * 2.0;
		b->positions[i*3] = (float)(len * sin((double)theta) * cos((double)fy));
		b->positions[i*3 + 1] = (float)(len * sin((double)theta) * sin((double)fy));
		b->positions[i*3 + 2] = (float)(len * cos((double)theta));
		
		 
	}
	for (int i = 0; i < num * 3; i++) {
		float r2 = (float)(rand()) / (float) (RAND_MAX);
		b->velocity[i] = VELOCITY_MIN + r2 * (VELOCITY_MAX - VELOCITY_MIN);
	}
	
}
