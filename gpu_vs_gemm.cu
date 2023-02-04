#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k, clock_t *time, clock_t *time1)
{ 
    *time = clock();
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
	    *time1 = clock();
            sum += a[row * n + i] * b[i * k + col];
	    *time1 = clock() - *time1;
        }
        c[row * k + col] = sum;
    }
    *time = clock() - *time;
} 


__global__ void gemm_matrix_mult(float* array1,  int rows1,  int cols1, float* array2,  int rows2, int cols2, float* array3,clock_t *time, clock_t *time1)
	{	
		*time = clock();
		//shared memory takes one tile at a time
		__shared__ float S1[TILE_WIDTH][TILE_HEIGHT];	//to store tiles for array 1
		__shared__ float S2[TILE_HEIGHT][TILE_WIDTH];	//to store tiles for array 2

		//threads x and y index for the current block
		unsigned int tx=threadIdx.x;	
		unsigned int ty=threadIdx.y;

		unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	//row value using x-index of current thread
		unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;	//column value using y-index of current thread

		unsigned int idx=c*rows1+r;				//column major index, using row and column value
		
		float val=0;		//register to store multiplication result initialized to zero

		for(int m=0; m<1+((rows2-1)/TILE_WIDTH);m++)	//going over all tiles one by one, with each m
		{
			
			*time1 = clock();
			int var1=m*TILE_WIDTH+tx ;		//x thread value for current tile
			int var2=m*TILE_WIDTH+ty ;		//y thread value for current tile
			
			//copying a tile from array1
			if (r < rows1 && var1 < rows2)		//if the value is associated to a valid matrix coordinate in array1 then store it to shared memory S1
				S1[ty][tx]=array1[r + var1*rows1];//storing a "valid" value from array to shared memory
			else
					S1[ty][tx]=0;					//storing zero, since there is no valid value
       			__syncthreads();						//syncing all threads once shared memory S1 is stored
			
			//copying a tile from array2
	       		if(c < cols2 && var2 < rows2)	//if value is associates to a valid matrix coordinate in array2 then store it to shared memory S2
	      			S2[ty][tx]=array2[var2+rows2*c];	//storing the valid value
	      		else 
	      			S2[ty][tx]=0;		//storing zero, since no valid value
			__syncthreads();		//synchronizing threads
			

			for(int i=0; i<TILE_WIDTH;i++)	//going over entire tile, ty row in S1 and tx column in S2
				val+=S1[ty][i]*S2[i][tx];	//and multiplying elements
			__syncthreads();		//synchronizing threads

			 *time1 = clock() - *time1;
		}
		
		if(r < rows1 && c< cols2)	//removing degenerate cases
			array3[idx]=val;	//saving multiplication result to global memory
			
		*time = clock() - *time;
	}

//generates matrix of size n by m with z rows of zero. Takes in already allocated matrix.

void matrix_generator(int m, int n, int z, int* matrix) {
	for (int i = 0; i < z; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i*n + j] = 0;
		}
	}
	for (int i = z; i < m; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i * n + j] = rand() % 1024;
		}
	}
}

void print_matrix(int m, int n, int* matrix) {
	for(int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d ", matrix[i * n + j]);
		}
		printf("\n");
	}
}


int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("Enter type in m n and k:\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM
    int *h_a, *h_b, *h_c, *h_cc, *h_cs;

    clock_t *gpu_time,*gemm_time, *time1, *time2;

    gpu_time = (clock_t *)malloc(sizeof(clock_t));
    gemm_time = (clock_t *)malloc(sizeof(clock_t));
    time1 = (clock_t *)malloc(sizeof(clock_t));
    time2 = (clock_t *)malloc(sizeof(clock_t));

    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    
    
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cs, sizeof(int)*m*k);

    int z; //stores # of 0s
    printf("Enter number of zero rows:\n");
    scanf("%d", &z);
    matrix_generator(m, n, z, h_a);

    printf("Matrix 1\n");
    print_matrix(m, n, h_a);

    matrix_generator(n, k, z, h_b);
    printf("Matrix 2\n");
    print_matrix(n, k, h_b);

    //ADDED SCANF and COMMENTED OUT LOOP TO TEST MY MATRIX FUNCTION
    /*
    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    */

    //float gpu_elapsed_time_ms;

    // some events to count the execution time
   // cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

	//time measurement using clock()
	clock_t *d_gpu_time;
	clock_t *d_gemm_time;
	clock_t *d_time1;
	clock_t *d_time2;
	
    // start to count execution time of GPU version
    //cudaEventRecord(start, 0);

    // Allocate memory space on the device 
    float *d_a, *d_b, *d_c, *d_cc, *d_cs;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);
    cudaMalloc((void **) &d_cc, sizeof(float)*m*k);
    cudaMalloc((void **) &d_cs, sizeof(float)*m*k);
    
    cudaMalloc((void **)&d_gpu_time, sizeof(clock_t));
    cudaMalloc((void **) &d_gemm_time, sizeof(clock_t));
    cudaMalloc((void **)&d_time1, sizeof(clock_t));
    cudaMalloc((void **) &d_time2, sizeof(clock_t));

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   

    //finding the clock rate of the GPU device 0
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float clock_rate = (prop.clockRate) * 1.0;
    printf("\nClockrate:%f\n",clock_rate);


    // Launch kernel GPU function 
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k,d_gpu_time, d_time1);  
    
    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_time, d_gpu_time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(time1, d_time1, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    // time counting terminate
   /* cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
*/
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %fms.\n\n", m, n, n, k, (*gpu_time/clock_rate)*1000.0);
    printf("Time for computing value for single element: %fms\n\n",(*time1/clock_rate)*1000.0);


    //computation using the gemm
    gemm_matrix_mult<<<dimGrid, dimBlock>>>(d_a, m, n, d_b, n, k, d_cc,d_gemm_time,d_time2);
   
    // Transfer results from device to host
    cudaMemcpy(h_cc, d_cc, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(gemm_time, d_gemm_time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(time2, d_time2, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GEMM: %fms.\n\n", m, n, n, k, (*gemm_time/clock_rate)*1000.0);
    printf("Time for computing value for single element: %fms\n\n",1000.0*(*time2/clock_rate));


    //computaion using the cublass for GEMM
	cublasHandle_t handle;
	cublasCreate(&handle);


    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	clock_t start1, end1;
	float alpha = 1.0;
	float beta = 0.0;

	start1 = clock();
	
	cudaEventRecord(start, 0);//cudaevent time starts
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha,d_a, m, d_b, n, &beta, d_cs, m);
	cudaEventRecord(stop, 0);//cudaevent time stop
	end1 = clock();
	cudaMemcpy(h_cs, d_cs, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
	printf("Time elapsed measured using clock() for SGEMM multiplication: %ld\n\n",end1-start1);

	// compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
     printf("Time elapsed measured using cudaEventRecord for SGEMM multiplication: %fms\n\n",gpu_elapsed_time_ms);
	cublasDestroy(handle);


    // validate results computed by GPU and GEMM
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j] != h_cs[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!\n");
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_cc);
    cudaFree(d_cs);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaFreeHost(h_cs);
    return 0;
}
