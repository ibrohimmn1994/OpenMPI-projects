#include "stencil.h"


int main(int argc, char **argv) {
	if (4 != argc) {
		printf("Usage: stencil input_file output_file number_of_applications\n");
		return 1;
	}
	char *input_name = argv[1];
	char *output_name = argv[2];
	int num_steps = atoi(argv[3]);
	MPI_Status status[4];
	MPI_Request requests[4];
	MPI_Datatype block;// vector for the local arrays
	MPI_Datatype extent;// vectore for the tails
	

        int rank, size, tag, tag1, tag2;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int before = (rank-1+size)%size;
        int after = (rank+1)%size;
        int len; // we should distributre the num_values to get the len which will be length of the local array in stencil application

	// Read input file
	double *input;
	int num_values;
        double *output;

	double *self_input;
	double *self_output;
	int k,i,j,s;
	double result, result1, result2, result3, result4;
	double *timings;	
	double MAX_EXECUTION_TIME;
        //#####################################################
        // Fetch the data
        if (rank == 0){

        	if (  ( 0 > (num_values = read_input(input_name, &input)) ) ) {
	        	return 2;
	        }
              
		

		if (NULL == (output = malloc(num_values * sizeof(double)))){
			perror("Couldn't allocate memory for output");
			return 2;
		}	
		timings = (double*)malloc(size *sizeof(double));
        }
	//####################################################
        MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
       
        len = num_values/size;
	
        self_output =(double*)malloc( (len+4)*sizeof(double));
	self_input = (double*)malloc( (len+4)*sizeof(double));
	double h = 2.0*PI/num_values;
	
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};
	
	//####################################################
	// Defie vectore : easier for communication
	MPI_Type_vector(len, 1, 1, MPI_DOUBLE, &block);
	MPI_Type_commit(&block);
//##########################################################################
//################# Input distribution
//####### This is one way to distribute the input, however scatter is nicer

	MPI_Scatter(&input[rank*len], 1, block, &self_input[EXTENT], 1, block, 0, MPI_COMM_WORLD );

	tag1 =1; tag2 = 2;

       //-----------------------------------------------------------------
//###### Instead posting a non-blocking recv before the send wnsures that no deadlock is possible

	//Start time
	double *send_left = (double*)malloc(EXTENT * sizeof(double));
	double *recv_left = (double*)malloc(EXTENT * sizeof(double));
	double *send_right = (double*)malloc(EXTENT * sizeof(double));
	double *recv_right = (double*)malloc(EXTENT * sizeof(double));
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	for ( s = 0; s < num_steps; s++  ){
		//Communicate the tail
		//##########################################################
		for (i=0; i<EXTENT; i++ ){
			send_left[i] = self_input[i+EXTENT];
			send_right[i] = self_input[len+i];
		}
		//##########################################################

		MPI_Irecv(recv_left, EXTENT, MPI_DOUBLE, before, tag1, MPI_COMM_WORLD, &requests[0]);
		MPI_Irecv(recv_right, EXTENT, MPI_DOUBLE, after, tag2, MPI_COMM_WORLD, &requests[1]);

		MPI_Isend(send_left, EXTENT, MPI_DOUBLE, before, tag2, MPI_COMM_WORLD, &requests[2]);

		MPI_Isend(send_right, EXTENT, MPI_DOUBLE, after, tag1, MPI_COMM_WORLD, &requests[3]);

	
		//##########################################################
		// Apply stencil on interior
		for ( i = EXTENT; i<len-EXTENT; i++ ){
			result = 0;
			for ( j =0; j < STENCIL_WIDTH; j++){
				result += STENCIL[j] * self_input[j+i];
			}
			self_output[i+EXTENT] = result;
		}
		//###########################################################
	
	//	MPI_Irecv(recv_left, EXTENT, MPI_DOUBLE, before, tag, MPI_COMM_WORLD, &requests[2]);
	//	MPI_Irecv(recv_right, EXTENT, MPI_DOUBLE, after, tag, MPI_COMM_WORLD, &requests[3]);
		MPI_Waitall(4, requests, status);// MPI_STATUS_IGNORE);
		//##########################################################
		for (i=0; i<EXTENT; i++){
			self_input[i] = recv_left[i];
			self_input[i+len+EXTENT] = recv_right[i];
		}
	
		//#########################################################
		// Apply stencil on the boundary
		result1=0;result2=0;result3=0;result4=0;
		for (j=0; j< STENCIL_WIDTH; j++){
			result1 += STENCIL[j] *self_input[j];
			result2 += STENCIL[j] *self_input[j+ len-1];
			result3 += STENCIL[j] *self_input[j+1];
			result4 += STENCIL[j] *self_input[j+ len-2];		
		}
		self_output[EXTENT] = result1;
		self_output[len-1+EXTENT] = result2;
		self_output[1+EXTENT] = result3;
		self_output[len-2+EXTENT] = result4;	
		
		
		//#########################################################


		if (s < num_steps  ){
			double *temp = self_input;
			self_input = self_output;
			self_output = temp;
		}
		MPI_Barrier(MPI_COMM_WORLD);

	}
	//#########################################################
	//Stop Time
	double my_execution_time = MPI_Wtime() - start;


	//############################################################
        // Here we should collect all the self_output to the global output as well the time measurements

	MPI_Gather(&self_input[EXTENT], 1, block, &output[rank*len], 1, block, 0, MPI_COMM_WORLD);
	free(self_input); free(self_output);
	
	MPI_Gather(&my_execution_time, 1, MPI_DOUBLE, &timings[rank], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//#############################################################
	
	if (rank == 0){
#ifdef PRODUCE_OUTPUT_FILE
		if (0 != write_output(output_name, output, num_values)){
			free(output); free(input);
			return 2;
		}
#endif
		find_max(size, timings );
		free(output); free(input);
	}
	//##############################################################

	MPI_Finalize();

	return 0;
}

//####################################################################
// Function to print array 
void print(int n, double *arr, int rank){
	printf("printing array of length %d for rank %d\n", n, rank);
	int i;
	for (i =0; i<n; i++){
		printf("%f ", arr[i]);
	}
	printf("\n");
}
// function to find the max execution time done by processors
void find_max(int n, double *arr){
	double max = arr[0];
	int i;
	int k = 0;
	for (i=1; i<n; i++){
		if(arr[i] > max){
			max = arr[i];
			k = i;
		} 
	}
//	printf("Max execution time %f occured at process %d\n", max, k);
	printf("\n%f\n", max);
}
// Read file
int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i=0; i<num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)){
		perror("Warning: couldn't close input file");
	}
	return num_values;
}

//Write file
int write_output(char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}
