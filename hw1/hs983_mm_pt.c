/********************************************************
* hs983_mm_pt.c calculate the matrix multiplication
* in parallel threads.
*
* gcc hs983_mm_pt.c -o fn -lpthread
* Input: dim - dimension of matrice, read from CMD line
*        n_threads - number of threads, read from CMD line
*
*     ECE 5720  HW1
*     Hongliang Si (hs983@cornell.edu)
*     2/15/2019
*********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

int dim=3, n_threads = 1,CurCol = 0;
double **A,**x_T,**y_T;
pthread_mutex_t mutex_col;

void *Matrix_multi(void *thr_id) {
  int cRow, cCol,k;
  while(1){
    pthread_mutex_lock(&mutex_col);{
      cCol=CurCol;
         if (CurCol >= dim){
           pthread_mutex_unlock(&mutex_col);
           return(0);
         }
         CurCol++;
      pthread_mutex_unlock(&mutex_col);
    }
    //The navie method for matrice multiplication.
    for(cRow = 0; cRow < dim; cRow++){
      for(k = 0; k < dim; k++){
        y_T[cCol][cRow] += A[cRow][k] * x_T[cCol][k];
      }
    }
  }
}

int main(int argc, char*argv[]){
  //Determine the dimension and # of threads. Detect invalid CMD line input.
  int check=1;
  if(argc != 3){
    printf("Invalid argument! \nDimension of N: ");
    scanf("%d",&dim);
    printf("Number of threads: ");
    scanf("%d",&n_threads);
  }else{
    dim       = atoi(argv[1]);
    n_threads = atoi(argv[2]);
  }

  //Thread hanlder.
  pthread_t *threads;

  //Declare drand48 and set the seed.
  double drand48();
  srand48(1);

  //Set struct for timming. n for nano s for sec.
  struct timespec start,finish;
  int ntime, stime;

  // Allocate dynamic memory for matrice.
  int cRow, cCol;
  threads = (pthread_t*)malloc(sizeof(pthread_t)*n_threads);
  A   = (double**)malloc(sizeof(double)*dim);
  x_T = (double**)malloc(sizeof(double)*dim);
  y_T = (double**)malloc(sizeof(double)*dim);
  for(cRow = 0; cRow < dim; cRow++){
    A[cRow] = (double*)malloc(sizeof(double)*dim);
    for(cCol = 0; cCol < dim; cCol++){
      A[cRow][cCol] = drand48();
    }
  }

  //I'm declaring the transpose.
  for(cCol = 0; cCol < dim; cCol++){
    x_T[cCol] = (double*)malloc(sizeof(double)*dim);
    y_T[cCol] = (double*)malloc(sizeof(double)*dim);
    for(cRow = 0; cRow < dim; cRow++){
      x_T[cCol][cRow] = drand48();
    }
  }

  //Start timming.
  clock_gettime(CLOCK_REALTIME,&start);

  //Fork threads.
  int t;
  for(t=0; t<n_threads; t++){
    pthread_create(&threads[t], NULL, Matrix_multi, (void*) &t);
  }

  //Join threads.
  for(t=0; t<n_threads; t++){
    pthread_join(threads[t], NULL);
  }
  pthread_mutex_destroy(&mutex_col);

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

  pthread_exit(NULL);
}
