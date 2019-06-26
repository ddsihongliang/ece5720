/**************************************************************************
* To compile: mpicc -o timing hs983_timing.c -O3 -lm
* To run: mpirun -np 2 --hostfile single_node ./timing
* Hongliang Si (hs983@cornell.edu)
* 3/26/2019
*   This code will perform a MPI communication timing test.
*   The processor with mypid = 0 will send character messages of
*   "length" elements to the processor with mypid = 1 "REPS" times.
*   Upon receiving the message a message of identical size is sent
*   back, then the delay time can be measured and averaged.
**************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define REPS 1000  //set # of iterations
#define MAXLENGTH 1048586 //set maximum message length to 2^20+10.

int main(int argc,char *argv[]) {
  int i,n,length;
  char *inmsg,*outmsg;
  int mypid,mysize,s_char;
  double start,finish,time,bw;
  MPI_Status status;
  s_char = sizeof(char);

  /* Initialize MPI */
  MPI_Init(&argc,&argv);

  /* Get the size of the MPI_COMM_WORLD communicator group */
  MPI_Comm_size(MPI_COMM_WORLD,&mysize);

  /* Get my rank in the MPI_COMM_WORLD communicator group */
  MPI_Comm_rank(MPI_COMM_WORLD,&mypid);

 if (mysize != 2){
   /* Chech enciroment variables*/
   fprintf(stderr, "Error: Set environment variable MP_PROCS to 2\n");
   exit(1);
 }

 length = 1;
 /* Determine message memory siez*/
 inmsg  = (char *) malloc(MAXLENGTH*sizeof(char));
 outmsg = (char *) malloc(MAXLENGTH*sizeof(char));

  /* synchronize the processes */
  MPI_Barrier(MPI_COMM_WORLD);
  /* Task 0 processing */
  if (mypid == 0){
    while(length<=100000){
      time = 0.0;
      /* round-trip timing test */
      printf("\n\nDoing round trip test for:\n");
      printf("Message length = %d character(s)\n",length);
      printf("Message size   = %d Bytes\n",s_char*length);
      printf("Number of Reps = %d\n",REPS);

      start = MPI_Wtime();
      for (n=1; n<=REPS; n++){
        /* Send message to process 1 */
        MPI_Send(&outmsg[0],length,MPI_CHAR,1,0,MPI_COMM_WORLD);
        /* Now wait to receive the echo reply from process 1 */
        MPI_Recv(&inmsg[0],length,MPI_CHAR,1,0,MPI_COMM_WORLD,&status);
      }
      finish = MPI_Wtime();

      /* Calculate round trip time average and bandwidth, and print */
      time = finish - start;
      printf("*** Round Trip Avg = %f uSec\n", (time/REPS*1E6));
      bw = 2.0*REPS*s_char*length/time;
      printf("*** time per word: tw = %f nSec/Byte\n",(1/bw)*1E9);
      length *=2;
    }
  }
  /* Task 1 processing */
  if (mypid == 1){
    while(length<=100000){
      for (n=1; n<=REPS; n++){
        /* receive message from process 0 */
        MPI_Recv(&inmsg[0],length,MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
        /* return message to process 0 */
        MPI_Send(&outmsg[0],length,MPI_CHAR,0,0,MPI_COMM_WORLD);
      }
      length *=2;
    }
  }
  /* End MPI*/
  MPI_Finalize();
  exit(0);
}
