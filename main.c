
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

#define N 1024


void mul(int *A,int *B,int *C){
    int d;
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++){
          d=0;
          for (int k=0;k<N;k++)
                d+=A[i*N+k]*B[i*N+k];
          C[i*N+j] = d;
        }
}

void mul_par(int *A,int *B,int *C){
    int d;
    #pragma omp parallel for private(d)
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++){
          d=0;
          for (int k=0;k<N;k++)
                d+=A[i*N+k]*B[i*N+k];
          C[i*N+j] = d;
        }
}

void mul_optim_trans(int *A,int *B,int *E){
    int *D = (int *)malloc(sizeof(int)* N *N);
    int d;
    for(int i=0;i<N;i++)
     for (int j=0;j<N;j++)
             D[i*N+j] = B[j*N+i];
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++){
          d=0;
          for (int k=0;k<N;k++)
                d+=A[i*N+k]*D[i*N+k];
          E[i*N+j] = d;
        }
}

void mul_optim_trans_par(int *A,int *B,int *E){
    int *D = (int *)malloc(sizeof(int)* N *N);
    int d;
    for(int i=0;i<N;i++)
     for (int j=0;j<N;j++)
             D[i*N+j] = B[j*N+i];

    #pragma omp parallel for private(d)
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++){
          d=0;
          for (int k=0;k<N;k++)
                d+=A[i*N+k]*D[i*N+k];
          E[i*N+j] = d;
        }
}

void mul_optim_Tiling(int *A,int *B,int *E){
        int block_size =64;
        int d;
        //Outside Blocks
        for(int i0=0;i0<N/block_size ;i0++)
                for(int j0=0;j0<N/block_size;j0++)
                                // Inside the Block
                                 for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                                         for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                                                 d=0;
                                                 for(int k=0;k< N;k++)
                                                         d+=A[i1*N+k]*B[k*N+j1];
                                                 E[i1*N+j1]=d;
                                         }
}
void mul_optim_Tiling_par(int *A,int *B,int *E){
        int block_size =64;
        int d;
        //Outside Blocks
        #pragma omp parallel for private (d)
        for(int i0=0;i0<N/block_size ;i0++)
                for(int j0=0;j0<N/block_size;j0++)
                                // Inside the Block
                                 for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                                         for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                                                 d=0;
                                                 for(int k=0;k< N;k++)
                                                         d+=A[i1*N+k]*B[k*N+j1];
                                                 E[i1*N+j1]=d;
                                         }
}

void mul_optim_trans_Tiling(int *A,int *B,int *E){
        int block_size =32;
        int *D = (int *)malloc(sizeof(int)* N *N);
        int d;
       for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
                D[j*N+i] = B[i*N+j];

        //Outside Blocks
        for(int i0=0;i0<N/block_size ;i0++)
                for(int j0=0;j0<N/block_size;j0++)
                                // Inside the Block
                                 for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                                         for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                                                 d=0;
                                                 for(int k=0;k< N;k++)
                                                         d+=A[i1*N+k]*D[j1*N+k];
                                                 E[i1*N+j1] =d ;

                                         }


}

void mul_optim_trans_Tiling_par(int *A,int *B,int *E){
        int block_size =32;
        int *D = (int *)malloc(sizeof(int)* N *N);
        int d;
       for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
                D[j*N+i] = B[i*N+j];

        //Outside Blocks
        #pragma omp parallel for private(d)
        for(int i0=0;i0<N/block_size ;i0++)
                for(int j0=0;j0<N/block_size;j0++)
                                // Inside the Block
                                 for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                                         for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                                                 d=0;
                                                 for(int k=0;k< N;k++)
                                                         d+=A[i1*N+k]*D[j1*N+k];
                                                 E[i1*N+j1] =d ;

                                         }


}

void mul_optim_Tiling_3D_par(int *A,int *B,int *E){
    int block_size =32;
    int d;
    //Outside Blocks
#pragma omp parallel for private(d)
    for(int i0=0;i0<N/block_size ;i0++)
        for(int j0=0;j0<N/block_size;j0++)
            for(int k0=0;k0<N/block_size;k0++)
                // Inside the Block
                for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                    for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                            d=0;
                            for(int k=k0*block_size;k< k0*block_size+block_size;k++)
                                d+=A[i1*N+k]*B[k*N+j1];
                            E[i1*N+j1]=d;

                    }


}

void mul_optim_Tiling_3D(int *A,int *B,int *E){
    int block_size =32;
    int d;
    //Outside Blocks
    for(int i0=0;i0<N/block_size ;i0++)
        for(int j0=0;j0<N/block_size;j0++)
            for(int k0=0;k0<N/block_size;k0++)
                // Inside the Block
                for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                    for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                            d=0;
                            for(int k=k0*block_size;k< k0*block_size+block_size;k++)
                                d+=A[i1*N+k]*B[k*N+j1];
                            E[i1*N+j1]=d;

                    }


}

void mul_optim_trans_Tiling_3D(int *A,int *B,int *E){
    int block_size =32;
    int d;
    int *D = (int *)malloc(sizeof(int)* N *N);
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            D[i*N+j] = B[j*N+i];

    //Outside Blocks
    for(int i0=0;i0<N/block_size ;i0++)
        for(int j0=0;j0<N/block_size;j0++)
            for(int k0=0;k0<N/block_size;k0++)
                // Inside the Block

                for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                    for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                            d=0;
                            for(int k=k0*block_size;k< k0*block_size+block_size;k++)
                            d+=A[i1*N+k]*D[j1*N+k];

                            E[i1*N+j1]=d;
                    }


}

void mul_optim_trans_Tiling_3D_par(int *A,int *B,int *E){
    int block_size =32;
    int d;
    int *D = (int *)malloc(sizeof(int)* N *N);
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            D[i*N+j] = B[j*N+i];

    //Outside Blocks
    #pragma omp parallel for
    for(int i0=0;i0<N/block_size ;i0++)
        for(int j0=0;j0<N/block_size;j0++)
            for(int k0=0;k0<N/block_size;k0++)
                // Inside the Block

                for(int i1=i0*block_size;i1< i0*block_size+block_size;i1++)
                    for(int j1=j0*block_size;j1< j0*block_size+block_size;j1++){
                            d=0;
                            for(int k=k0*block_size;k< k0*block_size+block_size;k++)
                            d+=A[i1*N+k]*D[j1*N+k];

                            E[i1*N+j1]=d;
                    }


}

void init_mat(int *A,int *B){
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
        {
                A[i*N+j] = 1;
                B[i*N+j] = 2 ;
        }
}

void init_mat_0(int *A){
    for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
        {
                A[i*N+j] = 0;
        }
}

int  test_mul(int *C,int *E){

     for(int i=0;i<N;i++)
        for (int j=0;j<N;j++)
                if (C[i*N+j]!= E[i*N+j]) {
                        printf("error");
                        return 0;
                }
     return 1;

}

int  affich_mat(int *C){

     for(int i=0;i<N;i++){
        for (int j=0;j<N;j++)
               printf("%d       ",C[i*N+j]);
        printf("\n");
     }
     return 1;

}

int main(void) {
        int *A = (int *)malloc(sizeof(int)* N *N);
        int *B = (int *)malloc(sizeof(int)* N *N);
        int *C = (int *)malloc(sizeof(int)* N *N);
        int *E = (int *)malloc(sizeof(int)* N *N);


        double dtime;

        init_mat(A,B);
        // omp_set_num_threads(20);


        dtime = omp_get_wtime();
        mul(A,B,C);
        dtime = omp_get_wtime() - dtime;
        printf("le temps pris pour une multiplication simple: %f\n", dtime);

        dtime = omp_get_wtime();
        mul_par(A,B,C);
        dtime = omp_get_wtime() - dtime;
        printf("le temps pris pour une multiplication avec parallelization: %f\n\n", dtime);

        dtime = omp_get_wtime();
        mul_optim_trans(A,B,E);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec transposition: %f\n", dtime);

        dtime = omp_get_wtime();
        mul_optim_trans_par(A,B,E);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec transposition et parrallelization: %f\n\n", dtime);



        dtime = omp_get_wtime();
        mul_optim_Tiling(A,B,C);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling: %f\n", dtime);

         dtime = omp_get_wtime();
        mul_optim_Tiling_par(A,B,C);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et parrallelization: %f\n", dtime);


        dtime = omp_get_wtime();
        mul_optim_trans_Tiling(A,B,C);
         dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et transposition: %f\n", dtime);

         dtime = omp_get_wtime();
        mul_optim_trans_Tiling_par(A,B,C);
         dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et transposition et parrallelization: %f\n\n", dtime);

        //----#D
        dtime = omp_get_wtime();
        mul_optim_Tiling_3D(A,B,C);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling 3D: %f\n", dtime);

         dtime = omp_get_wtime();
        mul_optim_Tiling_3D_par(A,B,C);
        dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et parrallelization 3D: %f\n", dtime);


        dtime = omp_get_wtime();
        mul_optim_trans_Tiling_3D(A,B,C);
         dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et transposition 3D: %f\n", dtime);

         dtime = omp_get_wtime();
        mul_optim_trans_Tiling_3D_par(A,B,C);
         dtime = omp_get_wtime() - dtime;
         printf("le temps pris pour une multiplication avec tiling et transposition et parrallelization 3D: %f\n", dtime);

    return 0;
}


