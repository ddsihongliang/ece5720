/********************************************************
* To Compile: gcc hs983_hw2_openmp_sort_block.c -fopenmp -O3 -o block
* This program sorts the diagonal line from biggest value and down to smallest
* 2D sorting
* To run: ./block n_row n_col num_thread
* Input: n_row - num of rows, read from CMD line
*       n_col - num of Cols
*      num_thread - num of threads
*   ECE 5720  HW2
*   Hongliang Si (hs983@cornell.edu)
*   3/9/2019
*********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define CHUNK_SIZE 1 //round robin
int *A,n_row,n_col, num_thread,dim;

void check_input(int argc,char*argv[]) {
  if (argc != 4){
    printf("Invalid argument! \nRows: Cols: Num_threads: \n");
  }
  else{
    n_row = atoi(argv[1]);
    n_col = atoi(argv[2]);
    num_thread = atoi(argv[3]);
  }
  if(n_col <= n_row) {dim = n_col;} // smaller value sets the dim.
  else {dim = n_row;}
}

//Swap matrix
void swap(int row_pos, int col_pos, int cur_pos) {
  int k;
  //Fork to swap the rows
  for(k = 0;k < n_row; k++)
  {
    int swap,cRow;
    cRow = cur_pos;   // Current working row
    swap = *(A + k + cRow*n_col);
    *(A + cRow*n_col    + k) = *(A + row_pos*n_col + k);
    *(A + row_pos*n_col + k) = swap;
  }

  //Fork to swap the columns
  for(k=0;k<n_col;k++)
  {
    int swap,cCol;
    cCol = cur_pos;   // Current working column
    swap = *(A + k*n_row + col_pos);
    *(A + k*n_row + col_pos) = *(A + k*n_row + cCol);
    *(A + k*n_row + cCol   ) = swap;
  }
}

int main(int argc, char*argv[]){
  //Determine the dimension and detect invalid CMD line input
  check_input(argc,argv);

  srand(3); //Declare the seed for rand().

  //Set array of pointers, and initiate Row, Col counters.
  //Populate matrix.
  A = (int *)malloc(sizeof(int)*n_row*n_col);
  int cRow,cCol;
  for(cRow = 0; cRow < n_row; cRow++) {
    for(cCol = 0; cCol < n_col; cCol++) {
      *(A + cCol + cRow*n_col) = rand() % 100;
    }
  }

  //Set struct for timming. n for nano s for sec.
  struct timespec start,finish,middle;
    int ntime, stime;
  double tot_time;

  int max,col_pos,row_pos,k,cur_pos;
  //sorting, start timming.
  clock_gettime(CLOCK_REALTIME,&start);
  for (cur_pos = 0; cur_pos < dim-1; cur_pos++)
  {
    if(dim-cur_pos<num_thread) //Optimize thread number
      num_thread = dim-cur_pos;
    omp_set_num_threads(num_thread);

    max = 1; //re-set max value
    if(num_thread != 1) { //Sequencial when num_thread = 1.
      //Fork threads to search max
      #pragma omp parallel
      {
        int loc_max = 1, //re-set max value
        loc_pos[2]; //position
        int ID = omp_get_thread_num();
        if( ID ==0&&cur_pos==0)
          printf("num of threads: %d\n", omp_get_num_threads());// Check
        #pragma omp for schedule(static,CHUNK_SIZE)
        for(cRow = cur_pos; cRow < n_row; cRow++)
        {
          for(cCol=cur_pos; cCol < n_col; cCol++)
          {
            if(*(A + cCol + cRow*n_col) > loc_max)
            {
              loc_max    = *(A + cCol + cRow*n_col);
              loc_pos[0] = cCol;
              loc_pos[1] = cRow;
            }
          }
        }
        if(max < loc_max){
          #pragma omp critical
          { //Find globle max.
            if(max < loc_max){
              col_pos = loc_pos[0];
              row_pos = loc_pos[1];
              max = loc_max;
            }
          }
        }
      }
      swap(row_pos,col_pos,cur_pos);
    }
    else{ // Sequencial start.
      if(cur_pos==0)
        printf("Sequencial start.\n");
      int loc_max = 1, //re-set max value
      loc_pos[2]; //position
      for(cRow = cur_pos; cRow < n_row; cRow++)
      {
        for(cCol=cur_pos; cCol < n_col; cCol++)
        {
          if(*(A + cCol + cRow*n_col) > loc_max)
          {
            loc_max    = *(A + cCol + cRow*n_col);
            loc_pos[0] = cCol;
            loc_pos[1] = cRow;
          }
        }
      }
    swap(loc_pos[1],loc_pos[0],cur_pos);
    }
  }

  //Finish timming and calculate run time in sec and nano_sec.
  clock_gettime(CLOCK_REALTIME, &finish);
        ntime = finish.tv_nsec - start.tv_nsec;
        stime = (int)finish.tv_sec - (int) start.tv_sec;
        tot_time = ntime / 1.0e9 + stime;
  printf("Job finished. Dimention: %dx%d \nTime: %f sec\n",n_row,
          n_col, tot_time);

  return 0;
}
