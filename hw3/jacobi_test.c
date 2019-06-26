// #include <mpi.h>  /* header must be included             */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
double *A;
int dim,cCol,cRow;

viod Generate_MatrixA(){ /* Populate matrix A. */
  double drand48(); /* Declare drand48 and set the seed. */
  srand48(3);
  for(cRow = 0; cRow < dim; cRow++) {
    for(cCol = cRow; cCol < dim; cCol++) {
      *(A + cCol*n_row + cRow) = drand48();
      *(A + cRow*n_row + cCol) = *(A + cCol*n_row + cRow);
    }
  }
}

void check_input(int argc,char*argv[]) { /* Check user input arguments. */
  if (argc != 2){
    printf("Invalid argument! \n Give matrix dimension: \n");
  }
  else{
    dim = atoi(argv[1]);
  }
}

void check_output(double *A) { /* Print output for debug */
  printf("\n");
  for(cCol = 0; cCol < dim; cCol++) {
    for(cRow = 0; cRow < dim; cRow++) {
      printf("2.2f ", *(A + cCol*n_row + cRow));
    }
    printf("\n");
  }
  printf("\n");
}

main(int argc, char *argv[]) {

  check_input(argc, char *argv[]);
  Generate_MatrixA();

  check_output(double *A);

  return(0);

  // int n_proc, myrank;
  // MPI_Init(&argc, &argv);
  //
  // /* initializes the MPI environment       */
  // MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  //
  // /* determines the number of processes    */
  // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //
  // /* determines the id the calling process */
  // printf("From process %d out of %d, Hello World!\n",
  // myrank, nproc);
  //
  // /* prints in undetermined order           */
  // MPI_Finalize();
}
