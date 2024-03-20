



#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define PRODUCE_OUTPUT_FILE


int read_input(const char *file_name, double **arr);
int write_output(const char *file_name, const double *output, int len);

void QS(int* local_len, double **array, MPI_Comm comm, int *depth, int command, double** kept, double** recieved );

bool isPowerOfTwo(int x);

int compare(const void *a, const void *b);
void print(int len, double *arr, int rank);
void print_int(int len, int *arr, int rank);
void find_max(int len, double *arr);
