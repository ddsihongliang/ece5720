#ifndef BODY_H_
#define BODY_H_


struct Body {
	int size;
	float* positions;
	float* velocity;
	float* mass;
};

extern void initialize(Body* b, int num);
#endif
