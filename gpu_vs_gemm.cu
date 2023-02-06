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

__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k, unsigned long long int *time)
{ 
    *time = clock64();
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row + i*m] * b[i + col*n];
        }
        c[col * m + row] = sum;
    }
    *time = clock64() - *time;
} 


__global__ void gemm_matrix_mult(float* array1,  int rows1,  int cols1, float* array2,  int rows2, int cols2, float* array3, unsigned long long int *time)
	{	
		*time = clock64();
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

		}
		
		if(r < rows1 && c< cols2)	//removing degenerate cases
			array3[idx]=val;	//saving multiplication result to global memory
			
		*time = clock64() - *time;
	}


//generates matrix of size n by m with z_row rows of zero and z_col zero columns
void matrix_generator(int m, int n, int z_row, int z_col, float* matrix) {
	for (int i = 0; i < z_row; i++) {
		for (int j = 0; j < z_col; j++) {
		//	matrix[i + z_row*j] = 0;     //column major
			 matrix[i * z_col + j] = 0;	//row major
		}
	}
	for (int i = z_row; i < m; i++) {
		for (int j = z_col; j < n; j++) {
			matrix[i + m*j] = i+1; //rand() % 1024;
		}
	}
}

//printing the matrix
void print_matrix(int m, int n, float* matrix) {
	for(int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//printf("%f ", matrix[i + m*j]);	//column major
			printf("%f ", matrix[i * n + j]);	//row major
		}
		printf("\n");
	}
	printf("\n\n");
}


int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("Enter type in m n and k:\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM
    float *h_a, *h_b, *h_c, *h_cc, *h_cs;

    unsigned long long int *gpu_time,*gemm_time;

    gpu_time = (unsigned long long int*)malloc(sizeof(unsigned long long int));
    gemm_time = (unsigned long long int *)malloc(sizeof(unsigned long long int));

    cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
    
    
    cudaMallocHost((void **) &h_c, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_cs, sizeof(float)*m*k);

    int z_row_a, z_col_a, z_row_b, z_col_b;
    printf("\nEnter number of zero rows and columns for matrix A:\n");
    scanf("%d %d", &z_row_a, &z_col_a);
    printf("\nEnter number of zero rows and columns for matrix B:\n");
    scanf("%d %d", &z_row_b, &z_col_b);

    //generating matrix
    matrix_generator(m, n, z_row_a, z_col_a, h_a);
    matrix_generator(n, k, z_row_b,z_col_b, h_b);

   /* //printing the matrices
    printf("\n\nMatrix A:\n");
    print_matrix(m, n, h_a);
    printf("\n\nMatrix B:\n");
    print_matrix(n, k, h_b);
*/

	//time measurement using clock()
	unsigned long long int *d_gpu_time;
	unsigned long long int *d_gemm_time;
	
	
    // start to count execution time of GPU version
    //cudaEventRecord(start, 0);

    // Allocate memory space on the device 
    float *d_a, *d_b, *d_c, *d_cc, *d_cs;
    cudaMalloc((void **) &d_a, sizeof(float)*m*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*k);
    cudaMalloc((void **) &d_c, sizeof(float)*m*k);
    cudaMalloc((void **) &d_cc, sizeof(float)*m*k);
    cudaMalloc((void **) &d_cs, sizeof(float)*m*k);
    
    cudaMalloc((void **)&d_gpu_time, sizeof(unsigned long long int));
    cudaMalloc((void **) &d_gemm_time, sizeof(unsigned long long int));
    

    // copy matrix A and B from host to device memory
    int err = cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}
    err = cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}

    //verifying the copy
  /*  err = cudaMemcpy(h_c, d_a, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}
    err = cudaMemcpy(h_cc, d_b, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}
    printf("\n\nMatrix A:\n");
    print_matrix(m, n, h_c);
    printf("\n\nMatrix B:\n");
    print_matrix(n, k, h_cc);
  */
    
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    //finding the device id
    int device;
    cudaGetDevice(&device);
    //printf("\nDevice:%d\n\n",device);

    //finding the clock rate of the GPU device 0
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    float clock_rate = (prop.clockRate) * 1.0;
   // printf("\nClockrate:%f\n\n",clock_rate);


    // Launch kernel GPU function 
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k,d_gpu_time);  
    
    // Transefr results from device to host 
    err = cudaMemcpy(h_c, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}
    err = cudaMemcpy(gpu_time, d_gpu_time, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    if(err > 0){printf("\nError copying: %d\n",err);return 0;}

    cudaDeviceSynchronize();

    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %fms.\n\n", m, n, n, k, (*gpu_time/clock_rate)*1000.0);


    //computation using the gemm
    gemm_matrix_mult<<<dimGrid, dimBlock>>>(d_a, m, n, d_b, n, k, d_cc,d_gemm_time);
   
    // Transfer results from device to host
    cudaMemcpy(h_cc, d_cc, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(gemm_time, d_gemm_time, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GEMM: %fms.\n\n", m, n, n, k, (*gemm_time/clock_rate)*1000.0);


    //computaion using the cublass for GEMM
	cublasHandle_t handle;
	cublasCreate(&handle);


    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const float alpha = 1.0;
    const float beta = 0.0;

	
    cudaEventRecord(start, 0);//cudaevent time starts
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, d_a, m, d_b, n, &beta, d_cs, m);

    cudaEventRecord(stop, 0);//cudaevent time stop
    cudaMemcpy(h_cs, d_cs, sizeof(float)*m*k, cudaMemcpyDeviceToHost);

	// compute time elapse on GPU computing
     cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
     printf("Time elapsed measured using cudaEventRecord for SGEMM multiplication: %fms\n\n",gpu_elapsed_time_ms);
     cublasDestroy(handle);
     
    // print_matrix(m,k,h_cs);

    // validate results computed by GPU and GEMM
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
	   // printf("GPU: %f, Gemm: %f, SGemm: %f\n", h_c[i + j * m],  h_cc[i + j * m],  h_cs[i + j * m]);	//column major
           // printf("GPU: %f, Gemm: %f, SGemm: %f\n", h_c[i*k + j],  h_cc[i*k + j],  h_cs[i*k + j]);	//row major
	   
	    if(h_cc[i + j * m] != h_c[i + j * m] && h_cc[i + j * m] != h_cs[i + j * m])	//column major	
	   // if(h_cc[i*k + j] != h_c[i*k + j] && h_cc[i*k + j] != h_cs[i*k + j])		//row major
	    {
                all_ok = 0;
		break;
            }
        }
       // printf("\n");
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
