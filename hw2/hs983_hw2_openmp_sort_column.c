/********************************************************
* To Compile: gcc hs983_hw2_openmp_sort_column.c -fopenmp -O3 -o column
* This program sorts the conlum from biggest value and down to smallest
* 1D sorting
* To run: ./column n_row n_col num_thread
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

int main(int argc, char*argv[]){
  //Determine the dimension and detect invalid CMD line input
  check_input(argc,argv);

  srand(3); //Declare the seed for rand().

  //Set array of pointers, and initiate Row, Col counters.
  //Populate matrix.
  int *A = (int *)malloc(sizeof(int)*n_row*n_col);
  int cRow,cCol,max,pos,k;

  for(cRow = 0; cRow < n_row; cRow++) {
    for(cCol = 0; cCol < n_col; cCol++) {
      *(A + cCol*n_row + cRow) = rand() % 100;
    }
  }

  //Set struct for timming. n for nano s for sec.
  struct timespec start,finish;
    int ntime, stime;
  double tot_time;

  //sorting, start timming.
  clock_gettime(CLOCK_REALTIME,&start);
  for(cRow = 0; cRow < dim; cRow++) {
    max = 1;
    if(num_thread != 1) { //Sequencial when num_thread = 1.
      //Reduce num_thread if remaining data size < threads
      if(dim-cRow<num_thread)
        num_thread = dim-cRow;
      omp_set_num_threads(num_thread);
      //Fork threads to search max
      #pragma omp parallel
      {
        int loc_max = 1, //re-set max value
        loc_pos; //position
        int ID = omp_get_thread_num();
        if( ID ==0&&cRow==0)
          printf("num of threads: %d\n", omp_get_num_threads());// Check
        #pragma omp for schedule(static)
          for(cCol=cRow; cCol < n_col; cCol++) {
            if(*(A + cCol*n_row + cRow) > loc_max) {
              loc_max = *(A + cCol*n_row + cRow);
              loc_pos = cCol;
            }
          }
        if(max < loc_max){
          #pragma omp critical
          { //Find globle max.
            if(max < loc_max){
              pos = loc_pos;
              max = loc_max;
            }
          }
        }
      }

      //Swap row
      for(k=0;k<n_row;k++)
      {
        int swap;
        // Current working column index is same as row
        cCol = cRow;
        swap = *(A + cCol*n_row + k);
        *(A + cCol*n_row + k) = *(A + pos*n_row + k);
        *(A + pos*n_row + k) = swap;
      }
    }
    else {
      //Sequencial when num_thread = 0.
      for(cCol=cRow; cCol < n_col; cCol++)
      {
        if(cRow==0&&cCol==0)
          printf("Sequencial Job started\n");
        if(*(A + cCol*n_row + cRow) > max)
        {
          if(*(A + cCol*n_row + cRow) > max) {
            max = *(A + cCol*n_row + cRow);
            pos = cCol;
          }
        }
      }
      //Swap the rows
      for(k=0;k<n_row;k++)
      {
        int swap;
        cCol = cRow;   // Current working column index is same as row
        swap = *(A + cCol*n_row + k);
        *(A + cCol*n_row + k) = *(A + pos*n_row + k);
        *(A + pos*n_row + k) = swap;
      }
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
