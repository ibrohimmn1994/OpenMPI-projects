


//#ifndef _ASSIGNMENT2
//#define _ASSINGMENT2

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cblas.h>
#define PRODUCE_OUTPUT_FILE

int read_input(const char *file_name, double **A, double **B);
int write_output(const char *file_name, const double *output, int dim);

void print(int n, double *matrix,int rank);
void find_max(int n, double *mat);

//#endif
