#include "CG1.h"
#define MAX 200
#define TOL 0.000001
int main(int argc, char **argv){
	if (2 != argc){
		printf("Usage: CG1 n");
		return 1;
	}
	
	
	int n = atoi(argv[1]);
	double h = 1./(n-1);
	int right, left, long_local_N;
	int size, rank, tag, i, j, N, local_N, local_n; 
	double time;
	double *u, *timings;
	double *local_b;
	
	double q0, local_q0, q1, local_q1, denominator, local_denominator, tau, beta;
	int iteration = 0;
	double *local_g, *local_u, *temp;
	//##########################################

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// This new communicator was created in attempt for implementing checkerboard algorithm, it is unnecessary now but makes no harm so I left it
	MPI_Comm new_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &new_comm);
	int new_rank, new_size;
	MPI_Comm_size(new_comm, &new_size);
	MPI_Comm_rank(new_comm, &new_rank);
	
	//if (new_size != size){MPI_Abort(new_comm, 1);}
	

	N = n*n;
	local_n = n/size;
	local_N = n * local_n;
	right = new_rank + 1; if (new_rank +1 >= new_size){right = MPI_PROC_NULL;}
	left = new_rank -1; if (new_rank -1<0){left = MPI_PROC_NULL;}
	long_local_N = local_N + 2*n;

	double *shared_memory;
	MPI_Win win;
	MPI_Win_allocate_shared(long_local_N*sizeof(double), 1, MPI_INFO_NULL, new_comm, &shared_memory, &win);

	//################################################################
	
	if (rank == 0){
		
		
		if (N%new_size != 0){
			printf("The number of processors should dividend of the dimension size");
			return 2;
		}
		u = malloc(N * sizeof(double));
		timings = malloc(size * sizeof(double));
	}
	
	local_g = malloc(local_N * sizeof(double));//bloc_Residual_Vecor
	local_u = malloc(local_N * sizeof(double));//bloc_vecot_x
	temp = malloc(local_N * sizeof(double));//buffer

	//######################################################################
	//      ( Check here )
	double *local_d = shared_memory;


//	local_d_next = shared_memory + long_local_N;
	//printf("%d:\n", new_rank);
	double x=0, y=0; int ind=0;
	for (i=0; i<local_n; i++){
		x = (i+rank*local_n)*h;
	//	printf("%f ", x);
		for (j=0; j<n; j++){
			y = j*h;
		//	printf("%f ", y);
			local_d[n+ind] = 2*h*h*( x*(1-x) + y*(1-y));
			ind++;
		}
	//	printf("\n");
	}
	//######################################################################
	



	double *left_ptr, *right_ptr;
	MPI_Aint adress;
	int disp_unit;
	MPI_Win_shared_query(win, right, &adress, &disp_unit, &right_ptr);
	MPI_Win_shared_query(win, left, &adress, &disp_unit, &left_ptr);
//	right_ptr += long_local_N;
//	left_ptr += long_local_N;

	//#####################################################################

	
	
	
	for (i=0; i<local_N; i++){
		local_g[i] = -local_d[n + i];//
		local_u[i] = 0;
	}
//	print_array(local_N, local_g, rank);
//	print_array(long_local_N, local_d, rank);	


	local_q0 = Norm(local_N, local_g, local_g, 0);
	MPI_Allreduce(&local_q0, &q0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//	printf("(%f, %f)\n", local_q0, q0);

	//######################################################################
	MPI_Win_lock_all(0, win);
	time = MPI_Wtime();	
//	if (new_rank ==0){printf("q0: %f\n", q0);}
	do{
	//	printf("%d,", iteration);
		MPI_Win_sync(win);
		MPI_Barrier(new_comm);

		if (right != MPI_PROC_NULL){
			for (i=0; i<n; ++i){
				local_d[i+long_local_N-n] = right_ptr[i+n];
			}
		}
		if (left != MPI_PROC_NULL){
			for (i=0; i<n; ++i){
				local_d[i] = left_ptr[i+long_local_N-2*n];
			}
		}
//		if (rank ==1 ){print_as_mesh(long_local_N,n, local_d, rank);}
		stencil_application(local_N, local_n, n, local_d, temp);
//		if (rank ==1){print_as_mesh(local_N,n,temp, rank);}
	
		//MPI_Allgather(local_d, local_N, MPI_DOUBLE, d, local_N, MPI_DOUBLE, MPI_COMM_WORLD);
	
		
		local_denominator = Norm(local_N, temp, local_d, n);
		MPI_Allreduce(&local_denominator, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		tau = q0/denominator;
		//printf("(%f, %f)\n", denominator, local_denominator);	
		
		for (i=0; i<local_N; i++){
			local_u[i] = local_u[i] + tau*local_d[n+i];
			local_g[i] = local_g[i] + tau*temp[i];
			temp[i] = 0;
		}
		
		local_q1 = Norm(local_N, local_g, local_g, 0);
	 	MPI_Allreduce(&local_q1, &q1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		beta = q1/q0;
		
		for (i=0; i<local_N; i++){
			local_d[i+n] = -local_g[i] + beta*local_d[i+n];
		}

		q0 = q1;
//	 	if (rank==0){	printf("%f\n", q0);;}
		
		iteration ++;
//		if (rank == 0){
//		printf("q0: %f\n", q0);}
	}while( (iteration < MAX));// &&(q0>TOL)  );
	MPI_Win_unlock_all(win);

	

	if (rank ==0 ){
		printf("\n\niterations: %d\nresidual: %.15f\n ", iteration, sqrt(q0));}


	MPI_Gather(local_u, local_N, MPI_DOUBLE, u, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD );

	
	time = MPI_Wtime() - time;

	MPI_Barrier(MPI_COMM_WORLD);

//	printf("%f\n", time);
	//######################################################################
	MPI_Gather(&time, 1, MPI_DOUBLE, &timings[rank], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	if (rank == 0){
		find_max(size, timings);

		free(u); free(timings);
	}

	free(local_u); 
	free(local_g); free(temp); 
	
	MPI_Win_free(&win);
	MPI_Comm_free(&new_comm);
	MPI_Finalize();
	return 0;
}

//##############################################################################
void print_as_mesh(int local_N, int n, double *array, int rank){
	printf("rank: %d\n", rank);
	int i;
	for (i=0; i<local_N; i++){
		if (i%n == 0){
			printf("\n");
		}
		printf("%f ", array[i]);
		
	}
	printf("\n");
}

void print_array(int n, double *array, int rank){
	printf("rank:%d\n", rank);
	int i;
	for (i=0; i<n; i++){
		printf(" %f ", array[i]);
	}
	printf("\n");
}

void find_max(int n, double *array){
	double max = array[0];
	int i;
	int k=0;
	for (i=1; i<n; i++){
		if (array[i] > max){
			max = array[i];
			k = i;
		}
	}
	printf("\ntime :%f\n", max);
}

//##############################################################################
double Norm(int n, double *array1, double *array2, int start ){
	int i;
	double norm = 0;
	for (i=0; i<n; i++){
		norm += array1[i]*array2[start+i];
	}
	return norm;
}

void stencil_application(int local_N, int local_n, int n, double *local_d, double *temp){
	int i, k;
	
	for (i=0; i<local_N; i++){
		temp[i] = 4*local_d[n+i] - (local_d[i] + local_d[n+i+n] + local_d[n+i+1] + local_d[n+i-1]);
	}
	// This is for the left and right boundary of the n*n mesh as the right boundary points have not right neighbours
	// and the left boundary points have no left neighbours.i

	for (i=0; i< local_n; i++){
		k = (i+1)*n-1;
		temp[k] += local_d[k +n+1];//right boundary of a row
		temp[k-n+1] += local_d[k-n];//left boundary of a row
	}	

}


//##############################################################################

