/**
 * mpi-episdet: Multi-GPU and multi-node CUDA application for high-order exhaustive epistasis detection using K2 scoring.
 *
 * Contact: Ricardo Nobre <ricardo.nobre@inesc-id.pt>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_combination.h>
#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include "combination.h"

#define MAX_LINE_SIZE 134217728	
#define MAX_NUM_LINES 1000000

#define NUM_COMB_PER_KERNEL_INSTANCE 131072
#define WORKGROUP_SIZE 32


unsigned long long * d_datasetCases;
unsigned long long * d_datasetControls;
uint * d_combinationArray;
float * d_lgammaPrecalc;
float * d_output;
int * d_output_index;


/* Reduces scores of combinations of SNPs
 * Based on: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler
 */
__inline__ __device__ float shfl_xor_32(float scalarValue, const int n)
{
	return __shfl_xor_sync(0xFFFFFFFF, scalarValue, n);
}

__inline__ __device__ float warpReduceMin(float val)
{
	val = min(val, shfl_xor_32(val, 1));
	val = min(val, shfl_xor_32(val, 2));
	val = min(val, shfl_xor_32(val, 4));
	val = min(val, shfl_xor_32(val, 8));
	val = min(val, shfl_xor_32(val, 16));
	return val;
}


__inline__ __device__ float blockReduceMin(float val)
{
	static __shared__ float shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMin(val);  
	if (lane==0) shared[wid]=val; 

	__syncthreads();              

	val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : FLT_MAX;
	if (wid==0) shared[0] = warpReduceMin(val);

	__syncthreads();

	val = shared[0];
	return val;
}


/* Saves best cadidate score
 * Based on: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
 */
__inline__ __device__ void atomicMin_g_f(float *addr, float val)
{
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = min(expected.f32,  val);
		current.u32  = atomicCAS( (unsigned int *)addr,
				expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}


/* Saves indexes of set of SNPs with best cadidate score */
__inline__ __device__ void atomicGetIndex(float *addr, float val, int *index)
{
	int next, expected, current;
	current = *index;

	do {
		expected = current;
		float global_minVal = *addr;
		if(val <= global_minVal) {	
			next = blockDim.x * blockIdx.x + threadIdx.x;
			current  = atomicCAS( index,
					expected, next);
		}
	} while( current != expected );
}


/* Evaluates a set of SNPs from combinationArray */
__global__ void epistasis(unsigned long long *datasetCases, unsigned long long *datasetControls, uint * combinationArray, float *lgammaPrecalc, float *output, int *output_index, int epistasisSize, int numSNPs, int casesSize, int controlsSize, ulong numCombinations, int comb)  
{
	int combination_i = blockDim.x * blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;	
	int i;
	int cases_i, controls_i;
	float score = FLT_MAX;

	extern __shared__ ushort smem[];
	ushort * observedValues_shared = smem;

	const int pow_table[10] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683};

	for(i=0;i<(WORKGROUP_SIZE * COMB_SIZE);i=i+WORKGROUP_SIZE) {       
		observedValues_shared[i + local_id] = 0;
		observedValues_shared[WORKGROUP_SIZE * COMB_SIZE + i + local_id] = 0;
	}
	__syncthreads();

	if(combination_i < numCombinations) {       

		score = 0;

		uint combination_table[EPISTASIS_SIZE];

		for(i=0; i < EPISTASIS_SIZE; i++) {

			combination_table[i] = combinationArray[numCombinations * i + combination_i];
		}

		unsigned long long casesArr[3 * EPISTASIS_SIZE];		
		unsigned long long controlsArr[3 * EPISTASIS_SIZE];        	

		for(cases_i = 0; cases_i < casesSize; cases_i++) {

			casesArr[0] = datasetCases[0 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];
			casesArr[1] = datasetCases[1 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];
			casesArr[2] = datasetCases[2 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];

			unsigned long long mask = (casesArr[0] | casesArr[1] | casesArr[2]);

			for(int epistasis_i=1; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {

				casesArr[epistasis_i * 3 + 0] = datasetCases[0 * numSNPs * casesSize + cases_i * numSNPs + combination_table[epistasis_i]];
				casesArr[epistasis_i * 3 + 1] = datasetCases[1 * numSNPs * casesSize + cases_i * numSNPs + combination_table[epistasis_i]];
				casesArr[epistasis_i * 3 + 2] = mask & ~(casesArr[epistasis_i * 3 + 0] | casesArr[epistasis_i * 3 + 1]); 
			}

			for(int comb_i = 0; comb_i < COMB_SIZE; comb_i++) {
				unsigned long long acc = 0xFFFFFFFFFFFFFFFF;

				for(int epistasis_i=0; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {
					acc = acc & casesArr[epistasis_i * 3 + ((int) (comb_i / pow_table[epistasis_i])) % 3];
				}

				observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 1 * WORKGROUP_SIZE + local_id] += __popcll(acc);
			}
		}

		for(controls_i = 0; controls_i < controlsSize; controls_i++) {

			controlsArr[0] = datasetControls[0 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];
			controlsArr[1] = datasetControls[1 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];
			controlsArr[2] = datasetControls[2 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];

			unsigned long long mask = (controlsArr[0] | controlsArr[1] | controlsArr[2]);

			for(int epistasis_i=1; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {

				controlsArr[epistasis_i * 3 + 0] = datasetControls[0 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[epistasis_i]];
				controlsArr[epistasis_i * 3 + 1] = datasetControls[1 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[epistasis_i]];
				controlsArr[epistasis_i * 3 + 2] = mask & ~(controlsArr[epistasis_i * 3 + 0] | controlsArr[epistasis_i * 3 + 1]); 
			}

			for(int comb_i = 0; comb_i < COMB_SIZE; comb_i++) {
				unsigned long long acc = 0xFFFFFFFFFFFFFFFF;
				for(int epistasis_i=0; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {
					acc = acc & controlsArr[epistasis_i * 3 + ((int) (comb_i / pow_table[epistasis_i])) % 3]; 
				}
				observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 0 * WORKGROUP_SIZE + local_id] += __popcll(acc);
			}
		}

		for (i=0; i< COMB_SIZE; i++) {        
			ushort zerosCount = observedValues_shared[i * 2 * WORKGROUP_SIZE + 0 * WORKGROUP_SIZE + local_id];
			ushort onesCount = observedValues_shared[i * 2 * WORKGROUP_SIZE + 1 * WORKGROUP_SIZE + local_id];

			score = score + __ldg(&lgammaPrecalc[zerosCount]) + __ldg(&lgammaPrecalc[onesCount]) - __ldg(&lgammaPrecalc[zerosCount + onesCount + 1]);
		}
		score = fabs(score);
	}

	float min_score =  blockReduceMin(score);        

	if(local_id == 0) {
		atomicMin_g_f(output, min_score);
	}

	if(score == min_score) {
		atomicGetIndex(output, min_score, output_index);
	}

}


void cuda_mem_init(int datasetCases_size, int datasetControls_size, int combinationArray_size, int lgammaPrecalc_size, int output_size)
{
	int ret = cudaMalloc((unsigned long long**)&d_datasetCases, datasetCases_size * sizeof(unsigned long long));
	ret = cudaMalloc((unsigned long long**)&d_datasetControls, datasetControls_size * sizeof(unsigned long long));
	ret = cudaMalloc((uint**)&d_combinationArray, combinationArray_size * sizeof(uint));
	ret = cudaMalloc((float**)&d_lgammaPrecalc, lgammaPrecalc_size * sizeof(float));
	ret = cudaMalloc((float**)&d_output, output_size * sizeof(float));
	ret = cudaMalloc((float**)&d_output_index, output_size * sizeof(int));
}


void cuda_mem_copy(unsigned long long* datasetCases, int datasetCases_size, unsigned long long* datasetControls, int datasetControls_size, uint* combinationArray, int combinationArray_size, float* lgammaPrecalc, int lgammaPrecalc_size, float* output, int output_size)
{
	int ret = 0;

	if(datasetCases != NULL)
		ret |= cudaMemcpy(d_datasetCases, datasetCases, datasetCases_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	if(datasetControls != NULL)
		ret |= cudaMemcpy(d_datasetControls, datasetControls, datasetControls_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	if(combinationArray != NULL)
		ret |= cudaMemcpy(d_combinationArray, combinationArray, combinationArray_size * sizeof(uint), cudaMemcpyHostToDevice);

	if(lgammaPrecalc != NULL)
		ret |= cudaMemcpy(d_lgammaPrecalc, lgammaPrecalc, lgammaPrecalc_size * sizeof(float), cudaMemcpyHostToDevice);

	if(output != NULL)
		ret |= cudaMemcpy(d_output, output, output_size * sizeof(float), cudaMemcpyHostToDevice);

	if(ret != 0)
		printf("There was a problem copying data to the GPU.\n");

}


void cuda_launch_kernel(int numSNPs, int controlsSize, int casesSize, int numCombinationsPerKernelInstance, int CpuThreadID, cudaStream_t stream_id)
{
	int blocksPerGrid = (size_t)ceil(((float)numCombinationsPerKernelInstance) / ((float)WORKGROUP_SIZE));	
	int comb = (int)pow(3.0, EPISTASIS_SIZE);

	epistasis<<<blocksPerGrid, WORKGROUP_SIZE, 2 * WORKGROUP_SIZE * comb * sizeof(ushort), stream_id>>>(d_datasetCases, d_datasetControls, d_combinationArray + (CpuThreadID * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE), d_lgammaPrecalc, d_output + CpuThreadID, d_output_index + CpuThreadID, EPISTASIS_SIZE, numSNPs, (int) ceil(casesSize / 64.0f), (int) ceil(controlsSize / 64.0f), numCombinationsPerKernelInstance, comb);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));

}

void cuda_clean_up()
{
	cudaFree(d_datasetCases);
	cudaFree(d_datasetControls);
	cudaFree(d_combinationArray);
	cudaFree(d_output);
	cudaFree(d_output_index);
}


/* Loads a sample from dataset in 0's, 1's and 2's format */
int getValues(char* line, u_char* data, u_char* data_target)
{
	int num = 0;
	const char* tok;
	for (tok = strtok(line, ",");
			tok && *tok;
			tok = strtok(NULL, ",\n"))
	{
		if(data != NULL) {
			data[num] = atoi(tok);
		}
		num++;
	}
	if(data_target != NULL) {
		data_target[0] = data[num - 1];
	}
	return num;
}


int main(int argc, char *argv[])
{
	int numSNPs;
	int sampleSize;
	int casesSize;
	int controlsSize;
	unsigned long long numCombinations;

	if(argc < 2) {
		printf("USE: infile\n");
		return 1;
	}

	struct timespec t_start, t_end;

	FILE* stream = fopen(argv[1], "r");	

	char * line = (char *) malloc(MAX_LINE_SIZE * sizeof(char));
	char * ret_fgets = fgets(line, MAX_LINE_SIZE, stream);
	int numCols = getValues(line, NULL, NULL);

	u_char * dataset = (u_char*) malloc(sizeof(u_char) * MAX_NUM_LINES * numCols);
	u_char * dataset_target = (u_char*) malloc(sizeof(u_char) * MAX_NUM_LINES);

	if(dataset == NULL) {
		printf("\nMemory allocation for dataset (genotype) failed.\n");
	}
	if(dataset_target == NULL) {
		printf("\nMemory allocation for dataset (phenotype) failed.\n");
	}

	sampleSize = 0;
	while (fgets(line, MAX_LINE_SIZE, stream))
	{
		getValues(line, dataset + (numCols * sampleSize), dataset_target + sampleSize);
		sampleSize++;
	}


	/* Counts the number of controls (0s) and cases (1s) */
	controlsSize = 0;
	casesSize = 0;
	for(int i=0; i < sampleSize; i++) {
		if(dataset_target[i] == 1) {

			casesSize++;
		}
		else {
			controlsSize++;
		}
	}

	int datasetOnes_64packed_size = ceil(((float) casesSize) / 64.0f) * (numCols - 1) * 3;
	unsigned long long * datasetOnes_64packed = (unsigned long long*) calloc(datasetOnes_64packed_size, sizeof(unsigned long long));

	int datasetZeros_64packed_size = ceil(((float) controlsSize) / 64.0f) * (numCols - 1) * 3;
	unsigned long long * datasetZeros_64packed = (unsigned long long*) calloc(datasetZeros_64packed_size, sizeof(unsigned long long));

	if(datasetOnes_64packed == NULL) {
		printf("\nMemory allocation for internal representation (cases) failed.\n");
	}
	if(datasetOnes_64packed == NULL) {
		printf("\nMemory allocation for internal representation (controls) failed.\n");
	}


	int numSamplesOnes_64packed = (int) ceil(((float) casesSize) / 64.0f);
	int numSamplesZeros_64packed = (int) ceil(((float) controlsSize) / 64.0f);

	/* Binarizes dataset */
	for(int column_j=0; column_j < (numCols - 1); column_j++) {     
		int numSamples0Found = 0;
		int numSamples1Found = 0;
		for(int line_i=0; line_i < sampleSize; line_i++) {

			int datasetElement = dataset[line_i * numCols + column_j];
			if(dataset_target[line_i] == 1) {
				int Ones_index = datasetElement * numSamplesOnes_64packed * (numCols - 1) + ((int)(numSamples1Found / 64.0f)) * (numCols - 1) + column_j;
				datasetOnes_64packed[Ones_index] = datasetOnes_64packed[Ones_index] | (((unsigned long long) 1) << (numSamples1Found % 64));
				numSamples1Found++;
			}

			else {
				int Zeros_index = datasetElement * numSamplesZeros_64packed * (numCols - 1) + ((int)(numSamples0Found / 64.0f)) * (numCols - 1) + column_j;
				datasetZeros_64packed[Zeros_index] = datasetZeros_64packed[Zeros_index] | (((unsigned long long) 1) << (numSamples0Found % 64));
				numSamples0Found++;
			}

		}
	}




        /* Initialize the MPI environment */
        MPI_Init(NULL, NULL);

        /* Get the number of processes */
        int mpi_world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

        /* Get the rank of the process */
        int mpi_world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);


        /* Get the name of the processor */
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);


	if(mpi_world_rank == 0) {
		printf("Using %d MPI processes.\n", mpi_world_size);
	}

	MPI_Barrier(MPI_COMM_WORLD); /* All nodes are ready */
        clock_gettime(CLOCK_MONOTONIC, &t_start);

	numSNPs = numCols - 1;
	numCombinations = mCk(numSNPs);

        if(mpi_world_rank == 0) {
		printf("#SNPs: %d\n", numSNPs);
		printf("#combinations: %llu\n", numCombinations);
                printf("sample size: %d\n#cases: %d, #controls:%d\n", sampleSize, casesSize, controlsSize);

	}


	int output_size = NUM_CPU_THREADS;

	cuda_mem_init(datasetOnes_64packed_size, datasetZeros_64packed_size, NUM_COMB_PER_KERNEL_INSTANCE * NUM_CPU_THREADS * EPISTASIS_SIZE, sampleSize, output_size);	

	float * outputFromGpu;
	cudaMallocHost((void **) &outputFromGpu, NUM_CPU_THREADS * sizeof(float));
	int * output_indexFromGpu;
	cudaMallocHost((void **) &output_indexFromGpu, NUM_CPU_THREADS * sizeof(int));

	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		outputFromGpu[i] = FLT_MAX;
	}

	float minScorePerCpuThread[NUM_CPU_THREADS];
	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		minScorePerCpuThread[i] = FLT_MAX;
	}

	unsigned long long indexMinScorePerCpuThread[NUM_CPU_THREADS];

	float * h_lgammaPrecalc = (float*) malloc(sampleSize * sizeof(float));
	if(h_lgammaPrecalc == NULL) {
		printf("\nMemory allocation for internal representation (cases) failed.\n");
	}

	/* Precalculates lgamma() values */
	for(int i=1; i < (sampleSize + 1); i++) {
		h_lgammaPrecalc[i - 1] = lgamma((double)i);
	}

	cuda_mem_copy(datasetOnes_64packed, datasetOnes_64packed_size, datasetZeros_64packed, datasetZeros_64packed_size, NULL, 0, h_lgammaPrecalc, sampleSize, outputFromGpu, output_size);	

	uint * combinationArray;
	cudaMallocHost((void**)&combinationArray, sizeof(uint) * NUM_COMB_PER_KERNEL_INSTANCE * NUM_CPU_THREADS * EPISTASIS_SIZE);

        cudaDeviceSynchronize();

        /* Explicitly create streams (1 per CPU thread) */
        cudaStream_t cudaStreamToUse[NUM_CPU_THREADS];
        for(int i=0; i < NUM_CPU_THREADS; i++) {
                cudaStreamCreate(&cudaStreamToUse[i]);
        }

	/* Each MPI process executes '#total_combinations / NUM_CPU_PROCESSES' */

	#pragma omp parallel for num_threads(NUM_CPU_THREADS) schedule(dynamic) 
	for(unsigned long long j = mpi_world_rank; j < (unsigned long long) ceil(numCombinations / (double) NUM_COMB_PER_KERNEL_INSTANCE); j += mpi_world_size) {

		int omp_thread_id = omp_get_thread_num();
		gsl_combination * c = gsl_combination_calloc (numSNPs, EPISTASIS_SIZE);

		int numCombToGenerate = min( (unsigned long long) NUM_COMB_PER_KERNEL_INSTANCE, numCombinations - ( j * NUM_COMB_PER_KERNEL_INSTANCE ));

		int startingCombination[EPISTASIS_SIZE];

		int retCombGen = combination(startingCombination, numSNPs, 1 + ( j * NUM_COMB_PER_KERNEL_INSTANCE ));
		if(retCombGen != 0) {
			printf("Problem in iteration: %llu, starting combination index: %lld\n", j, 1 + ( j * NUM_COMB_PER_KERNEL_INSTANCE ));
			continue;
		}

		/* Sets combination to the one that the cpu thread must start with */
		size_t * combData = gsl_combination_data(c);
		for(int z=0; z<EPISTASIS_SIZE; z++) {
			combData[z] = startingCombination[z];
		}

		for(int comb_i = 0; comb_i < numCombToGenerate; comb_i++) {
			for(int z=0; z<EPISTASIS_SIZE; z++) {
				combinationArray[(omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE) + z*numCombToGenerate + comb_i] = combData[z];
			}
			gsl_combination_next (c);
		}
		gsl_combination_free(c);

		int ret = cudaMemcpyAsync(d_combinationArray + (omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE), combinationArray + (omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE), numCombToGenerate * EPISTASIS_SIZE * sizeof(uint), cudaMemcpyHostToDevice, cudaStreamToUse[omp_thread_id]);

		outputFromGpu[omp_thread_id] = FLT_MAX;
		cudaMemcpyAsync(d_output + omp_thread_id, outputFromGpu + omp_thread_id, sizeof(float), cudaMemcpyHostToDevice, cudaStreamToUse[omp_thread_id]);

		cuda_launch_kernel(numSNPs, controlsSize, casesSize, numCombToGenerate, omp_thread_id, cudaStreamToUse[omp_thread_id]);

		cudaMemcpyAsync(outputFromGpu + omp_thread_id, d_output + omp_thread_id, sizeof(float), cudaMemcpyDeviceToHost, cudaStreamToUse[omp_thread_id]);
		cudaMemcpyAsync(output_indexFromGpu + omp_thread_id, d_output_index + omp_thread_id, sizeof(int), cudaMemcpyDeviceToHost, cudaStreamToUse[omp_thread_id]);

		cudaStreamSynchronize(cudaStreamToUse[omp_thread_id]);

		if(outputFromGpu[omp_thread_id] < minScorePerCpuThread[omp_thread_id]) {
			minScorePerCpuThread[omp_thread_id] = outputFromGpu[omp_thread_id];
			indexMinScorePerCpuThread[omp_thread_id] = j * NUM_COMB_PER_KERNEL_INSTANCE + output_indexFromGpu[omp_thread_id];
		}

	}

        cudaDeviceSynchronize(); 

	float minScore = FLT_MAX;
	unsigned long long indexOfMinScore;
	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		if(minScorePerCpuThread[i] < minScore) {
			minScore = minScorePerCpuThread[i];
			indexOfMinScore = indexMinScorePerCpuThread[i];
		}
	}






	/* Reduce scores and get indices of best score */

	float * minScorePerMpiProcess; 
        unsigned long long * indexOfMinScorePerMpiProcess;

    	if ( mpi_world_rank == 0) { 
       		minScorePerMpiProcess = (float *)malloc(mpi_world_size*1*sizeof(float)); 
                indexOfMinScorePerMpiProcess = (unsigned long long *)malloc(mpi_world_size*1*sizeof(unsigned long long));

       	} 

	MPI_Gather( &minScore, 1, MPI_FLOAT, minScorePerMpiProcess, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);	
	MPI_Gather( &indexOfMinScore, 1, MPI_UNSIGNED_LONG_LONG, indexOfMinScorePerMpiProcess, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);        


	float minScoreAllMpiProcesses = FLT_MAX;
	unsigned long long indexOfMinScoreAllMpiProcesses;

	if(mpi_world_rank == 0) {
		for(int i = 0; i < mpi_world_size; i++) {
			if(minScorePerMpiProcess[i] < minScoreAllMpiProcesses) {
				minScoreAllMpiProcesses = minScorePerMpiProcess[i];
				indexOfMinScoreAllMpiProcesses = indexOfMinScorePerMpiProcess[i];
			}
		}

	}


	int epistasisSize = 3;
        if(mpi_world_rank == 0) {

		printf("Printing best solution\n");
		int bestFoundCombination[ epistasisSize];
		combination(bestFoundCombination, numSNPs, 1 + indexOfMinScoreAllMpiProcesses);
		for(int comb_index=0; comb_index < epistasisSize; comb_index++) {
			printf("%d ", bestFoundCombination[comb_index]);
		}

		printf("%.2lf \n", minScoreAllMpiProcesses);

	}


	cuda_clean_up();

	free(dataset);  
	free(dataset_target);
	cudaFreeHost(outputFromGpu); 
	cudaFreeHost(output_indexFromGpu);
	cudaFreeHost(combinationArray);

	MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &t_end);

	MPI_Finalize();


	if(mpi_world_rank == 0) {
		double timing_duration_mpi = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
		printf("Epistasis search time:\t%0.3lf seconds\n", timing_duration_mpi);
		std::cout << "Tera unique sets per sec. scaled to sample size: " << std::fixed << std::setprecision(6) << (((double) numCombinations * (double) (sampleSize) / (double)(timing_duration_mpi)) / 1e12) << std::endl;
	}

	return 0;
}

