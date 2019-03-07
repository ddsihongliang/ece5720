/********************************************************
* hs983_mm_rbyc.c calculate the matrix multiplication
* in navie method.
*
* gcc hs983_mm_rbyc.c -O3
* Input: dim - dimension of matrice, read from CMD line
*
*     ECE 5720  HW1
*     Hongliang Si (hs983@cornell.edu)
*     2/15/2019
*********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char*argv[]){
  //Determine the dimension and detect invalid CMD line input
  int dim;
  if (argc != 2){
    printf("Invalid argument! \nDimension: ");
    scanf("%d",&dim);
  }else{dim = atoi(argv[1]);}

  //Declare drand48 and set the seed.
  double drand48();
  srand48(1);

  //Set array of pointers, and initiate Row, Col counters.
  //x_T means transpose.
  double *A[dim],*x_T[dim],*y_T[dim];
  int cRow,cCol,k;

  //Set struct for timming. n for nano s for sec.
  struct timespec start,finish;
  int ntime, stime;

  // Allocate dynamic memory for matrice.
  for(cRow = 0; cRow < dim; cRow++){
    A[cRow] = (double*)malloc(sizeof(double)*dim);
    for(cCol = 0; cCol < dim; cCol++){
      A[cRow][cCol] = drand48();
    }
  }
  //This will make sense when we do the calculation.
  //Remember that I'm declaring the transpose.
  for(cCol = 0; cCol < dim; cCol++){
    x_T[cCol] = (double*)malloc(sizeof(double)*dim);
    y_T[cCol] = (double*)malloc(sizeof(double)*dim);
    for(cRow = 0; cRow < dim; cRow++){
      x_T[cCol][cRow] = drand48();
    }
  }

  //Row-by-Column multiplication, start timming.
  clock_gettime(CLOCK_REALTIME,&start);
  for(cCol = 0; cCol < dim; cCol++){
    for(cRow = 0; cRow < dim; cRow++){
      for(k = 0; k < dim; k++){
        y_T[cCol][cRow] = y_T[cCol][cRow] + A[cRow][k] * x_T[cCol][k];
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
  printf("Job finished. Dimention: %d \nTime: %ld sec, %ld nsec\n", dim, stime,ntime);

  return 0;
}
