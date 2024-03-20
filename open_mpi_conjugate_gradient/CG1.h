
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



void print_as_mesh(int local_N, int n, double *array, int rank);
void print_array(int n, double *array, int rank);

void find_max(int n, double *array);


double Norm(int n, double *array1, double *array2, int start);

void stencil_application(int local_N, int local_n, int n, double *local_d, double *temp);

