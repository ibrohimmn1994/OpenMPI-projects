#include "quicksort.h"

int main(int argc, char **argv){
	if (4 != argc){
		printf("Usage: quicksort input_file output_file method_number");
	}

	char *input_name = argv[1];
	char *output_name = argv[2];
	int command = atoi(argv[3]);

	double *input_array;
	double *output_array;
	int size, rank, tag, len;
	double time;
	double *timings;
		
	int depth = 0 ;
	int indicator;
	int i;
	int *lengths;
	int *disp;
	int *counts;
	int *counts_disp;
	int c;


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm sub_com;
	MPI_Status status;

	double* kept =NULL;
        double* recieved =NULL;
	double *local_array;
	//######## Open the file and store data ###############################
	if (rank == 0){
		if ( (0 > (len = read_input(input_name, &input_array))) ){
			printf("Error in array length");
			return 2;
		}
		output_array = malloc(len * sizeof(double));
		timings = malloc(size * sizeof(double));
		lengths = malloc(size * sizeof(int));

		
	}
	MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//####################################################################
	/*
 * 	This code block is to limit the number of pe
 * 	if the number of Pe is bigger than the number of element
 * 	we find the closest number that is power of two and equal or smaller
 * 	than the number of elmenet. This number will be the number of Pe 
 * 	that we will use
 * 	
 * 	This is the case if the number of pe exceeds the number of elements
 * 	the whole code block from line 54 to 124 is to make the number of pe 
 * 	independent of n.
 	*/
	MPI_Group group;
	MPI_Comm_group(MPI_COMM_WORLD, &group);
	MPI_Group new_group;
	int limit = size;
	if (size > len){
		
		limit = len;
		while(!isPowerOfTwo(limit)){
			limit--;
		}
	}
	int range[limit];
	for (c=0; c<limit; c++){
		range[c] = c;
	}
	MPI_Group_incl(group, limit, range, &new_group);
	int group_ranks;
	MPI_Group_rank(new_group, &group_ranks);
	MPI_Comm MPI_COMM_NEW;
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &MPI_COMM_NEW);
	if (group_ranks == MPI_UNDEFINED){

		MPI_Finalize();
		exit(0);
	}
	MPI_Comm_size(MPI_COMM_NEW, &size);
	MPI_Comm_rank(MPI_COMM_NEW, &rank);
	//#####################################################################
	// Check if the number of processors is appropriate
	if ( !isPowerOfTwo(size)){
		printf("The number of processors should be power of 2");
		MPI_Finalize();
		return 2;
	}

	//#####################################################################
	/*
 * 	In the following code block concerns with distributing array's elements 
 * 	equally among the processes
 * 	so if we have array of length 10 and 4 pe. each pe gets 2 excpet every second pe gets 3
 * 	if we have array of length 10 and 8 pe. each pe gets 1 except every fourth one gets 2
 	*/
	/*
 	// The code below that decides the split size is lengthy and complicated
 	// however, it ensures that the ranks <size/2 get equal number of array
 	// elements with the ranks>size/2 => better work-balance
	 */
	// 
	int local_len;
	//local_array = malloc(len*sizeof(double));
	// if there is a reminder
	if ( (len/size) != ((double)len)/size){
		if (rank ==0){
			counts = malloc(size*sizeof(int));
			counts_disp = malloc(size *sizeof(int));
		}
		local_len = (int)len/size;
		// Note (Important)
		//  you said that my code for the split size is incorrect cause reminder can be zero.
		//   However reminder here is never zero because of
		//   the condition above  (len/size) != ((double)len)/size which
		//   means that the reminder is not zero
		
		int reminder = len - size*local_len;
		int size_increment = size/reminder;

		local_len += (rank%(size_increment))/(size_increment-1);
		

		MPI_Gather(&local_len, 1, MPI_INT, counts ,1, MPI_INT, 0, MPI_COMM_NEW);
		if (rank == 0){
			counts_disp[0] = 0;
			int sum_counts = 0;
			for (c=1; c<size; c++){
				sum_counts += counts[c-1];
				counts_disp[c] = sum_counts;
			}
			
		}
			
		local_array = malloc(local_len*sizeof(double));
		MPI_Scatterv(input_array, counts, counts_disp, MPI_DOUBLE, local_array, local_len, MPI_DOUBLE, 0, MPI_COMM_NEW);
		if (rank == 0){free(counts); free(counts_disp);}
	}
	// else if there isno reminder
	else{
		local_len = len/size;
		local_array = malloc(local_len*sizeof(double));
		
		MPI_Scatter(input_array, local_len, MPI_DOUBLE, local_array, local_len, MPI_DOUBLE, 0, MPI_COMM_NEW);
	}
	

	//#####################################################################
	//##### Start Sorting #################################################
	time = MPI_Wtime();
	qsort(local_array, local_len, sizeof(double), compare);

	QS(&local_len, &local_array, MPI_COMM_NEW, &depth, command, &kept, &recieved);
	time = MPI_Wtime() - time;

	//#####################################################################
	// Gather the number of elements in each processors at the end of the computation and compute the displacement vector. Both to be used in MPI_Gatherv()

	MPI_Gather(&local_len, 1, MPI_INT, lengths, 1, MPI_INT, 0, MPI_COMM_NEW);
	if (rank == 0){
		disp = malloc(size * sizeof(int));
		disp[0] = 0;
		int sum_length = 0;
		for (i=1; i<size; i++){
			sum_length += lengths[i-1];
			disp[i] = sum_length;
		} 
	}

	MPI_Gatherv(local_array, local_len, MPI_DOUBLE, output_array, lengths, disp, MPI_DOUBLE, 0, MPI_COMM_NEW);
	
	// Gather execution time of each processor
	MPI_Gather(&time, 1, MPI_DOUBLE, &timings[rank], 1, MPI_DOUBLE, 0, MPI_COMM_NEW);	
	//#######################################################################

	if (rank == 0){
		find_max(size, timings);
#ifdef PRODUCE_OUTPUT_FILE
		if (0 != write_output(output_name, output_array, len)){
			return 2;
		}
#endif

		free(output_array);
		free(input_array);
		free(timings);
		free(lengths);
		free(disp);
	}	
	free(local_array);
	free(kept);
	free(recieved);
	MPI_Finalize();


	return 0;
}
//##############################################################################
bool isPowerOfTwo(int x){
	return x && (!(x&(x-1)));
}

int compare(const void *d, const void *b){
	if (*(double*)d > *(double*)b){return 1;}
	else if(*(double*)d < *(double*)b){return -1;}
}

//##############################################################################
void QS(int* local_len, double **array, MPI_Comm old_comm,  int *depth, int command, double** kept, double** recieved){
	int rank, size;

	MPI_Status status, Status;
	MPI_Comm new_comm;
	MPI_Comm_size(old_comm, &size);
	MPI_Comm_rank(old_comm, &rank);

	MPI_Request request;
	
	double local_mean = 0;
	double mean = 0;
	double local_median;
	double *median_array;
	int total_len;
	double pivot;
	int count;
	int count_send, count_recv;
	int len_kept;
	int i,j, start, ind;

	*depth += 1;
	// Base case
	if (size == 1){
		return ;
	}
	//######## Find the pivot ###############
	/*
	// Using mean:
	if (command == 1){
		for (i = 0; i< *local_len; i++){
			local_mean += (*array)[i];
		}
		MPI_Reduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, old_comm);
		
		MPI_Reduce(local_len, &total_len,1, MPI_INT, MPI_SUM, 0, old_comm);
		if (rank == 0){
			pivot = mean/total_len;		
		}
	}
	*/

	if (command == 1){
		if (rank == 0){
			if (*local_len %2 == 0){
				pivot = ((*array)[*local_len/2] + (*array)[*local_len/2-1])/2;
			}
			else{
				pivot = (*array)[*local_len/2];
			}
			
		}
		MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, old_comm);
	}
	//---------------------------------------------------------------------

	if (command == 2){	
		if (*local_len %2 == 0){
			
			local_median = ((*array)[*local_len/2] + (*array)[*local_len/2-1])/2;
		}
		else{
			local_median = (*array)[*local_len/2];
		}
		median_array = malloc(size* sizeof(double));
		MPI_Allgather(&local_median, 1,MPI_DOUBLE,median_array, 1,MPI_DOUBLE, old_comm);
		qsort(median_array, size, sizeof(double), compare);
                pivot = (median_array[size/2] + median_array[size/2-1])/2;	
	}
	//---------------------------------------------------------------------
	if (command == 3){
		if (*local_len %2 == 0){
			local_median = ((*array)[*local_len/2] + (*array)[*local_len/2-1])/2;
		}
		else{
			local_median = (*array)[*local_len/2];
		}
		MPI_Allreduce(&local_median, &pivot, 1,MPI_DOUBLE, MPI_SUM, old_comm);
		pivot = pivot/size;
	}
	//######################################################################
	//######### Find the index of the pivot #######
	int pivot_ind = 0;
	for (i=0; i<*local_len; i++){
		if ((*array)[i] > pivot){
			
			pivot_ind = i-1;
			break;
		}
		
	}
	if (i == *local_len){pivot_ind = i-1;}
	
	//########### Exchange the value that are larger and smaller of the pivor
	// between the processors ###################################
	if (rank < size/2){
		count = *local_len-1-pivot_ind;
	}
	else{ count =pivot_ind + 1; }

	count_send = count;
	MPI_Sendrecv_replace(&count, 1, MPI_INT,  (rank+size/2)%size, 0, (rank+size/2)%size, 0, old_comm, &Status);
        
	count_recv=count;
	
	if (rank < size/2){
		

		len_kept = pivot_ind + 1;
//		MPI_Send(&count_send, 1, MPI_INT, rank+size/2, 1, old_comm);
		MPI_Send(*array+ len_kept, count_send, MPI_DOUBLE, rank+ size/2, 2, old_comm );
		 
		*recieved = (double*)realloc(*recieved, (count_recv+1) * sizeof(double));

//		 MPI_Isend(*array+len_kept, count_send,MPI_DOUBLE,rank+size/2,1, old_comm, &request);
//		MPI_Irecv(*recieved, count_recv, MPI_DOUBLE, rank+size/2,2, old_comm, &request );
		
		*kept = (double*)realloc(*kept, (len_kept+1) * sizeof(double));
		j = 0;
		
//		MPI_Wait( &request, &status);
		MPI_Recv(*recieved, count_recv, MPI_DOUBLE, rank + size/2, 4, old_comm, MPI_STATUS_IGNORE);
		
	}
	else{
	
	
		*recieved = (double*)realloc(*recieved, (count_recv+1) * sizeof(double));
		len_kept = *local_len-1-pivot_ind;
		MPI_Recv(*recieved, count_recv, MPI_DOUBLE, rank-size/2, 2, old_comm, MPI_STATUS_IGNORE);
		
	
		
//		MPI_Irecv(*recieved, count_recv, MPI_DOUBLE, rank-size/2,1, old_comm, &request );
//		 MPI_Isend(*array, count_send,MPI_DOUBLE,rank-size/2,2, old_comm, &request);

                *kept = (double*)realloc(*kept,(len_kept+1) * sizeof(double));
                j = pivot_ind + 1;
               // MPI_Wait( &request, &status);
		MPI_Send(*array, count_send, MPI_DOUBLE, rank-size/2, 4, old_comm);
	}

	// The values that are kept in the processor
	// Here, kept: are the value that kept in the processors either larger or smaller
	// than the pivot value and recieved are the value that recieved from the 
	// pair processor that are likewise smaller or larger of the pivot value
	// Then this two arrays will be merged together

	for (i=0; i<len_kept; i++){
		(*kept)[i] = (*array)[j];
		j++;	
	}

	//####################################################################
	//###########  Merging kept and recieved ###########################
	i=0; j=0; ind =0;
	//Merge (1)
//	while (i < count){
//		array[ind] = recieved[i];
//		i++;
//		ind++;
//	}
//	while (j < len_kept){
//		array[ind] = kept[j];
//		j++;
//		ind++;
//	}
//	qsort(count+len_kept, array, sizeof(double), compare);

	//Merge (2)
	*local_len = len_kept + count_recv;
	*array = (double*)realloc(*array, (*local_len+1)*sizeof(double) );

	while(i< count_recv && j < len_kept){
		if((*recieved)[i] < (*kept)[j]){
			(*array)[ind] = (*recieved)[i];
			i++;
		}
		else{
			(*array)[ind] = (*kept)[j];
			j++;
		}
		ind++;
	}
	if (i>=count_recv){
		while(j < len_kept){
			(*array)[ind] = (*kept)[j];
			j++;
			ind++;
		}
	}
	if (j>= len_kept){
		while(i< count_recv){
			(*array)[ind] = (*recieved)[i];
			i++;
			ind++;
		}
	}

	//#####################################################################
	//#################### Divide the processors into two groups and apply quicksort recuirsively###################################

	int color = rank/(size/2);
	MPI_Comm_split(old_comm, color, rank, &new_comm);


	QS( local_len, array, new_comm, depth, command,kept, recieved);

	return ;
}
//##########################################################################

int read_input(const char *file_name, double **arr){
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))){
		perror("Could not open input file");
		return -1;
	}
	int len;
	if (EOF == fscanf(file, "%d ", &len)){
		perror("Could not read element count");
		return -1;
	}

	if (NULL == (*arr = malloc(len * sizeof(double)))){
		perror("Could not allocate memory to the array");
	}

	for (int i=0; i<len; i++){
		if ( EOF == fscanf(file, "%lf", &((*arr)[i]) )){
			perror("Could not read elements into the array");
		}
	}

	if (0 != fclose(file)){
		perror("Could not close the input file");
	}
	return len;
}
//##############################################################################

int write_output(const char *file_name, const double *output, int len){
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))){
		perror("Erro in opening the output file");
		return -1;
	}
	for (int i=0; i<len; i++){
		if (0 > fprintf(file, "%9.f ", output[i])){
			perror("Could not write to putput file");
		}
	}
	if (0> fprintf(file, "\n")){
		perror("Could not write to output file");
	}
	if (0 != fclose(file)){
		perror("Could not close the output file");
	}

	return 0;

}
//##############################################################################

void find_max(int len, double *array){
	double max = array[0];
	int i;
	int k=0;
	for (i=1; i<len; i++){
		if(array[i] > max){
			max = array[i];
			k = i;
		}
	}
	printf("%f\n", max);
}
//##############################################################################
void print(int len, double *array, int rank){
	int i;
	printf("rank : %d\n", rank);
	for (i=0; i<len; i++){
		printf("%f ", array[i]);
	}
	printf("\n\n");
}
//##############################################################################
void print_int(int len,int* array, int rank){
	int i;
	printf("rank : %d\n", rank);
	for (i=0; i<len; i++){
		printf("%d ", array[i]);
	}
	printf("\n");
}
