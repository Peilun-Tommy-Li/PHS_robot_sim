//PHS_simulation
//Author: Dr.Beckers, modified by Tommy Li
//Date: Jul.10, 2023
//Description: the C code accelerate the comp. This function is called from "PHS_Kernel_func.py"
/*kernel
 *Kernel - Function for covariance
 To compile and use, use gcc/clang to compile this file to .o (object file), then compile the .o file to .so (shared lib)

 Open commend line, navigate to the directory that has this file
 type in:
 1. gcc -c -fPIC PHSKernel_se.c -o PHSKernel_se.o

 2. gcc -shared -o PHSKernel_se.so PHSKernel_se.o
 Then the simulation should be able to use
 */


#include <stdio.h>
#include <math.h>

void Product(double* x, double* y, double* z,int n,int m,int dim, double sd, double* l)
{
    int i, j, k;
    int count = 0;
    double dis;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            dis = 0;
            for (k = 0; k < dim; k++) {
                dis += pow((x[i * dim + k] - y[j * dim + k]) / l[k], 2);
                /* printf("x: %f, y: %f, k: %d\n", x[i * dim + k], y[j * dim + k], k); */
            }
            z[count] = sd * exp(-1 * dis / 2);
            count++;
        }
    }
}


void Derivation(double* x, double* y, double* z,int n,int m,int dim, double sd, double* l)
{
    int i, j, k;
    int count = 0;
    double dis, dis1;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            for (int d = 0; d < dim; d++) {
                dis = 0;
                for (k = 0; k < dim; k++) {
                    dis += pow((x[i * dim + k] - y[j * dim + k]) / l[k], 2);
                    // printf("x: %f, y: %f, k: %d\n", x[i * dim + k], y[j * dim + k], k);
                }
                if (d < dim)
                    dis1 = (x[i * dim + d] - y[j * dim + d]) / l[d] / l[d];
                else
                    dis1 = 2;
                z[count] = sd * exp(-1 * dis / 2) * dis1;
                count++;
            }
        }
    }
}


void DDerivation(double *x, double *y, double *z, int n,int m,int dim,double sd,double *l)
{
    int i,j,k;
    int count=0;
    double dis,dis1;
    for (i=0; i<n; i++) {
        for (int d2=0; d2<dim; d2++) {
    	for (j=0; j<m; j++) {
        for (int d1=0; d1<dim; d1++) {
            dis=0;
            for (k=0; k<dim; k++) {
                dis+=pow((x[i*dim+k]- y[j*dim+k])/l[k],2);
                // printf("x: %f, y: %f, k: %d\n",x[i*dim+k],y[j*dim+k],k);
            }
            if (d1<dim && d2<dim && d1!=d2) {
                dis1=  -(x[i*dim+d1]-y[j*dim+d1])/l[d2]/l[d2]*(x[i*dim+d2]- y[j*dim+d2])/l[d1]/l[d1];
            }
            else if (d1<dim && d2<dim && d1==d2) {
                dis1=  (1-(x[i*dim+d1]-y[j*dim+d1])/l[d2]/l[d2]*(x[i*dim+d2]- y[j*dim+d2]))/l[d1]/l[d1];
            }
            else {
                dis1=2;
            }
            z[count] = sd*exp(-1*dis/2)*dis1;
            count++;
        }
    }
}
}
}

void kernel(double* A, double* B, double hyp_sd, double* hyp_l, const int c, double* result, int rowA, int colA, int rowB, int colB, int rowHYP_l)
{
    double sd = pow(hyp_sd, 2);

    if (rowA != rowB || rowHYP_l != rowA)
    {
        printf("Matrix dimensions do not match");
        return;
    }

    if (c == 2) {
        DDerivation(A, B, result, colA, colB, rowA, sd, hyp_l);
    }
    else if (c == 1) {
        Derivation(A, B, result, colA, colB, rowA, sd, hyp_l);
    }
    else {
        Product(A, B, result, colA, colB, rowA, sd, hyp_l);
    }
}


