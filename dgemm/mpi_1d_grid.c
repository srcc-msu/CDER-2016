/*******************************************************************************
 * Copyright (c) 2016 SRCC MSU
 *
 * Author: Alexander Antonov <asa@parallel.ru>
 *
 * Distributed under the MIT License - see the accompanying file LICENSE.
 ******************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define N 4096

int main(int argc, char* argv[])
{
   int i, j, k, rank, size, rows, cols, myrows, mycols, last_rows, last_cols, myi, myj, l, m, jmax;
   double t1, t2, sum, s;

/* Initialization */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

/* Parameters calculation */
   myrows=mycols=rows=cols=(N-1)/size+1;
   last_rows=last_cols=N-rows*(size-1);
   if(rank==size-1) myrows=mycols=last_rows;

/* Memory allocation */

   double **a=(double**)malloc(sizeof(double*)*myrows);
   for(i=0; i<myrows; i++)
      a[i]=(double*)malloc(sizeof(double)*N);

   double **b=(double**)malloc(sizeof(double*)*N);
   for(i=0; i<N; i++)
      b[i]=(double*)malloc(sizeof(double)*mycols);

   double **c=(double**)malloc(sizeof(double*)*myrows);
   for(i=0; i<myrows; i++)
      c[i]=(double*)malloc(sizeof(double)*N);

   double *b0=(double*)malloc(sizeof(double)*N*cols);

   double **buf=(double**)malloc(sizeof(double*)*N);

   for(i=0; i<N; i++)
      buf[i]=b0+i*cols;

/* Matrix initialization */
   for(i=0; i<myrows; i++){
      myi=i+rank*rows;
      for(j=0; j<N; j++){
         a[i][j]=(myi+j)*0.0001e0;
         c[i][j]=0.0e0;
      }
   }

   for(i=0; i<N; i++)
      for(j=0; j<mycols; j++){
         myj=j+rank*cols;
         b[i][j]=(i-myj)*0.0001e0;
      }

   MPI_Barrier(MPI_COMM_WORLD);
   t1=MPI_Wtime();

/* Column groups  */
   for(m=0; m<size; m++){
      if(rank==m)
         for(j=0; j<mycols; j++)
            for(k=0; k<N; k++)
               buf[k][j]=b[k][j];

/* Sending of b matrix */
      l=cols*N;
      MPI_Bcast(b0, l, MPI_DOUBLE, m, MPI_COMM_WORLD);
      if(m==size-1) jmax=last_cols;
      else jmax=cols;

/* Calculation */
      for(i=0; i<myrows; i++)
         for(j=0; j<jmax; j++)
            for(k=0; k<N; k++)
               c[i][m*cols+j]+=a[i][k]*buf[k][j];
   }
   MPI_Barrier(MPI_COMM_WORLD);
   t2=MPI_Wtime();

/* Control sun calculation */
   for(i=0, sum=0.0e0; i<myrows; i++)
      for(j=0; j<N; j++)
         sum+=c[i][j];
   MPI_Reduce(&sum, &s, 1, MPI_DOUBLE, MPI_SUM, 0,
              MPI_COMM_WORLD);

   if(rank==0) printf("N= %d, Nproc=%d, Sum=%lf, Time=%lf\n", N, size, s, t2-t1);

   MPI_Finalize();
}
