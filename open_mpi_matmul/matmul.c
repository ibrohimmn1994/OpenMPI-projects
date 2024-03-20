#include "matmul.h"
//#include <cblas.h>
#include <gsl/gsl_blas.h>
int main(int argc, char **argv){
	if (3 != argc){
		printf("Usage: matmul input_file output_file");
		return 1;
	}
	
	char *input_name = argv[1];
	char *output_name = argv[2];
	double *inputA;
	double *inputB;
	double *outputC;
	double *A;
	double *B;
	double *C;
	double *A_extra;
	double *timings;

	int ORDER = 101;
	char TRANSA = 111;
	char TRANSB = 111;

	int size, rank, tag, dim, dim_local;
	double time;
	int i;
	int sum;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//----------------------------------------------------
	int CartSize = sqrt(size);
	if (size != CartSize*CartSize){
		if (rank == 0){
			printf("The Cartesian grid should be a square");
		}
		MPI_Finalize();
		return 2;
	}
	
	//---------------------------------------------------
	if (rank == 0){

		if ( (0 > (dim = read_input(input_name, &inputA, &inputB))) ){
			return 2;
		}	
		if (dim % CartSize != 0){
			printf("Matrix's dimension should be divisible by the Cartesian-grid's dimension");
			return 2;
		}
		outputC = malloc(dim*dim *sizeof(double));
		timings = malloc(size * sizeof(double));

	}

	MPI_Bcast(&dim, 1, MPI_INT, 0, 	MPI_COMM_WORLD);
	dim_local = dim/CartSize;
	A = malloc(dim_local*dim_local * sizeof(double));
	B = malloc(dim_local*dim_local * sizeof(double));
	C = malloc(dim_local*dim_local * sizeof(double));
	A_extra = malloc(dim_local*dim_local * sizeof(double));

	//---------------------------------------------------
	// Create Cartesian Grid for the processors:

	MPI_Comm CartComm;
	MPI_Comm ColComm;
	MPI_Comm RowComm;
	int CartDim[2];
	int CartPer[2];
	int CartCoor[2];
	int CartSub_row[2];
	int CartSub_col[2];
	CartDim[0] = CartSize; CartDim[1] = CartSize;
	CartPer[0] = 0       ; CartPer[1] = 0;

	MPI_Cart_create(MPI_COMM_WORLD, 2, CartDim, CartPer, 1, &CartComm);
	MPI_Cart_coords(CartComm, rank, 2, CartCoor);

	CartSub_row[0] = 0; CartSub_row[1] = 1;
	MPI_Cart_sub(CartComm, CartSub_row, &RowComm);

	CartSub_col[0] = 1; CartSub_col[1] = 0;
	MPI_Cart_sub(CartComm, CartSub_col, &ColComm);

	//----------------------------------------------------
	//Data distribution

	double *RowStrip_A = (double*)malloc(dim*dim_local * sizeof(double));
	double *RowStrip_B = (double*)malloc(dim*dim_local * sizeof(double));
	if (CartCoor[1] == 0){
		MPI_Scatter(inputA, dim*dim_local, MPI_DOUBLE, RowStrip_A, dim*dim_local, MPI_DOUBLE, 0, ColComm );
		MPI_Scatter(inputB, dim*dim_local, MPI_DOUBLE, RowStrip_B, dim*dim_local, MPI_DOUBLE, 0, ColComm );
	}

	for (i=0; i<dim_local; i++){
		MPI_Scatter(&RowStrip_A[i*dim],dim_local, MPI_DOUBLE, &(A_extra[i*dim_local]), dim_local, MPI_DOUBLE, 0, RowComm);
		MPI_Scatter(&RowStrip_B[i*dim], dim_local, MPI_DOUBLE, &(B[i*dim_local]), dim_local, MPI_DOUBLE, 0, RowComm);
	}
	
	free(RowStrip_A); free(RowStrip_B);


	//-----------------------------------------------------
	//Matrix-matrix multiplication

	int lead, j, after, before;
	int k1,k2,k3;
	double time2;
	after = (CartCoor[0]+1)%CartSize;
	before = (CartCoor[0]-1)%CartSize;

	time = MPI_Wtime();
	for (i=0; i<CartSize; i++){
		//Sending local A to the row processors
		lead = (CartCoor[0]+i)%CartSize;
		if (CartCoor[1] == lead){
			for(j=0; j<dim_local*dim_local; j++){
				A[j] = A_extra[j];
			}
		}
		MPI_Bcast(A, dim_local*dim_local, MPI_DOUBLE, lead, RowComm);

		// Multiplication of local matrices
//		for (k1=0; k1<dim_local*dim_local; k1++){
//			C[k1] += A[k1]*B[k1];
//		}

		//######################################################
		// ((  with BLAS )) gives 26.158118 for n=3600 and 4 PEs

		/*	
		cblas_dgemm(ORDER, TRANSA, TRANSB, dim_local, dim_local, dim_local, 1, A,dim_local , B,dim_local ,1,C,dim_local);
		*/

		//########################################################
		// ((  ijk-method )) gives 166 for n=3600 and 4 PEsi

		/*
		double store;
		for(k1=0; k1<dim_local; k1++){
			for(k2=0; k2<dim_local; k2++){
				store = 0;
				for(k3=0; k3<dim_local; k3++){
					store += A[k1*dim_local+k3]*B[k3*dim_local+k2];
				}
				C[k1*dim_local+k2] = store;		
			}
		}
		*/
		//######################################################
		// ((  kij-method )) gives 57s for n=3600 and 4 PEs
		// This one is the fastest loop order
		
		double store;
		for (k3=0; k3<dim_local; k3++){
			for (k1=0; k1<dim_local; k1++){
				store = A[k1*dim_local+k3];
				for (k2=0; k2<dim_local; k2++){
					C[k1*dim_local+k2] += store*B[k3*dim_local+k2];
				}
			}
		}
		
		

		// #####################################################
		// (( jik-method ))

		/*
		double store;
		for (k2=0; k2< dim_local; k2++){
			for (k1=0; k1<dim_local; k1++){
				store = 0;
				for (k3=0; k3<dim_local; k3++){
					store += A[k1*dim_local+k3] * B[k3*dim_local+k2];
				}
				C[k1*dim_local+k2] = store;
			}
		}
		*/
	

		MPI_Sendrecv_replace(B, dim_local*dim_local, MPI_DOUBLE, after, 0, before, 0, ColComm, &status);
	}



	time = MPI_Wtime() - time;

	MPI_Gather(&time, 1, MPI_DOUBLE, &timings[rank], 1, MPI_DOUBLE, 0 , MPI_COMM_WORLD );	

	//--------------------------------------------------------
	// Data collection

	double *RowResults = (double*)malloc(dim*dim_local * sizeof(double));
	for (i=0; i<dim_local; i++){
		MPI_Gather(&C[i*dim_local], dim_local, MPI_DOUBLE, &RowResults[i*dim], dim_local, MPI_DOUBLE, 0, RowComm);
	}
	if (CartCoor[1] == 0){
		MPI_Gather(RowResults, dim*dim_local, MPI_DOUBLE,outputC, dim*dim_local, MPI_DOUBLE, 0, ColComm);
	}

	free(RowResults);

	//-----------------------------------------------------------
	if (rank == 0){
		find_max(size, timings);
#ifdef PRODUCE_OUTPUT_FILE
		if (0 != write_output(output_name, outputC, dim)){
			return 2;	
		}
#endif
	}
	if (rank == 0){free(outputC); free(inputA); free(inputB);}
	free(A); free(B); free(C); free(A_extra);
	MPI_Finalize();
	return 0;
}

//############################################################################








int read_input(const char *file_name, double **A, double **B){
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))){
		perror("Could not open input file");
		return -1;
	}
	int num;
	if (EOF == fscanf(file, "%d ", &num)){
		perror("Could not read element count from input file");
		return -1;
	}
	
	if (NULL == (*A = malloc(num*num * sizeof(double)))){
		perror("Could not allocate memory to A");
		return -1;
	}
	if (NULL == (*B = malloc(num*num * sizeof(double)))){
		perror("Could not allocate memory to B");
		return -1;
	}

	
	for (int i=0; i<num*num; i++){
		if (EOF == fscanf(file, "%lf",&((*A)[i]) )){
			perror("Could read element for A");
			return -1;
		}

	}
	
	for (int j=0; j<num*num; j++){
		if (EOF == fscanf(file, "%lf", &((*B)[j]))){
			perror("could not read elements for B");
			return -1;
		}
	}

	if (0 != fclose(file)){
		perror("Could not close input file");
	}
	return num;
}

void print(int num, double *matrix, int rank){
	int i;
	printf("rank: %d\n", rank );
	for(i=0; i<num*num; i++){
		if (i%num == 0){
			printf("\n");
		}
		printf("%f ", matrix[i]);
	}
	printf("\n\n");
}

void find_max(int n, double *mat){
	double max = mat[0];
	int i;
	int k=0;
	for (i=1; i<n; i++){
		if (mat[i]>max){
			max = mat[i];
			k=i;
		}
	}
	printf("%f\n", max);
}
int write_output(const char *file_name, const double *output, int num){
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))){
		perror("Could not open output file");
		return -1;
	}
	for (int i=0; i< num*num; i++){
		if (0 > fprintf(file, "%.6f ", output[i])){
			perror("Could not write to putput file");
		}
	}
	if (0> fprintf(file, "\n")){
		perror("Could not write to output file");
	}
	if (0 != fclose(file)){
		perror("could not close the output file");
	}
	return 0;
}
