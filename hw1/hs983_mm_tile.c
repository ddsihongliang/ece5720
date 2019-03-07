/********************************************************
* hs983_mm_tile.c calculate the matrix multiplication
* in tile (block) method.
*
* gcc hs983_mm_tile.c -O3
* Input: dim - dimension of matrice, read from CMD line
*        tile - size of blocks, read from CMD line
*
*     ECE 5720  HW1
*     Hongliang Si (hs983@cornell.edu)
*     2/15/2019
*********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int bool;
#define true 1
#define false 0

int main(int argc, char*argv[]){
  //Determine the dimension and check for invalid CMD line input.
  int dim=3, tile=3, check=1;
  while(check){
    if(argc != 3 || dim%tile != 0){
      if(dim%tile != 0){printf("Dimension of N has to be divisible by tile!\n");}
      printf("Invalid argument! \nDimension of N: ");
      scanf("%d",&dim);
      printf("Dimension of tiles: ");
      scanf("%d",&tile);
    }else{
      dim  = atoi(argv[1]);
      tile = atoi(argv[2]);
    }
    if(dim%tile == 0){check = 0;}
  }

  //Declare drand48 and set the seed.
  double drand48();
  srand48(1);

  //Set array of pointers, and initiate Row, Col counters.
  double *A[dim],*x[dim],*y[dim];
  int i,j,k,b,n,m;

  //Set struct for timming. n for nano s for sec.
  struct timespec start,finish;
  int ntime, stime;

  // Allocate dynamic memory for matrice.
  for(i = 0; i < dim; i++){
    A[i] = (double*)malloc(sizeof(double)*dim);
    y[i] = (double*)malloc(sizeof(double)*dim);
    for(j = 0; j < dim; j++){
      A[i][j] = drand48();
    }
  }
  for(i = 0; i < dim; i++){
    x[i] = (double*)malloc(sizeof(double)*dim);
    for(j = 0; j < dim; j++){
      x[i][j] = drand48();
    }
  }

  //Block multiplication, start timming.
  clock_gettime(CLOCK_REALTIME,&start);
  for(m = 0;m < dim;m += tile){
    for(n = 0;n < dim;n += tile){
      for(b = 0;b < dim;b += tile){
        for(i = 0; i < tile; i++){
          for(j = 0; j < tile; j++){
            for(k = 0; k < tile; k++){
              y[i+m][j+n] += A[i+m][k+b] * x[j+n][k+b];
            }
          }
        }
      }
    }
  }
  //Finish timming and calculate run time in sec and nano_sec.
  clock_gettime(CLOCK_REALTIME,&finish);
  if(finish.tv_nsec < start.tv_nsec){
    ntime = 1000000000 + finish.tv_nsec - start.tv_nsec;
    stime = (int) finish.tv_sec - (int) start.tv_sec - 1;
  }else{
    ntime = finish.tv_nsec - start.tv_nsec;
    stime = (int) finish.tv_sec - (int) start.tv_sec;
  }
  printf("Job finished. Dimention: %d \nTime: %ld sec, %ld nsec\n", dim, stime, ntime);

  return 0;
}
