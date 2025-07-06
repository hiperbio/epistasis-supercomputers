/**
 *
 * mpi-Epi4Tensor: Multi-GPU and multi-node epistasis detection specialized to fourth-order searches on modern GPU microarchitectures with matrix processing cores
 *
 * High-Performance Computing Architectures and Systems (HPCAS) Group, INESC-ID
 * Contact: Ricardo Nobre <ricardo.nobre@inesc-id.pt>
 *
 */

/* Standard Library */
#include <iostream>
#include <iomanip>      
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
#include <libgen.h>	
#include <mpi.h>

#include "cutlass-op.hpp"
#include "epistasis.hpp"
#include "reduction.hpp"

#define MAX_CHAR_ARRAY 1000

#define NUM_GPUS 1

/* Used for allocating arrays of binary data */
typedef typename cutlass::Array<cutlass::uint1b_t, 32> ScalarBinary32;


/* Reads number of SNPs, number of controls/cases and controls/cases file name. */
int readDatasetDescriptionFile(const char *fileName, uint *numSNPs, char *controlsFileName, uint *numControls, char *casesFileName, uint *numCases) {

	FILE *fStream = fopen(fileName, "r");                	// File with information and pointers to dataset.
	if(fStream == NULL) {
		std::cerr << "File '" << fileName << "' does not exist!" << std::endl;
		return 1;
	}

	char line[MAX_CHAR_ARRAY];
	char *ret = fgets(line, MAX_CHAR_ARRAY, fStream);       // First line represents the number of SNPs.
	*numSNPs = atoi(line);

	ret = fgets(controlsFileName, MAX_CHAR_ARRAY, fStream); // Second line represents the filename with controls data.
	controlsFileName[strcspn(controlsFileName, "\n")] = 0;  // Removes trailing newline character.

	ret = fgets(line, MAX_CHAR_ARRAY, fStream);             // Third line represents the number of controls.
	*numControls = atoi(line);

	ret = fgets(casesFileName, MAX_CHAR_ARRAY, fStream);    // Forth line represents the filename with cases data.
	casesFileName[strcspn(casesFileName, "\n")] = 0;        // Removes trailing newline character.

	ret = fgets(line, MAX_CHAR_ARRAY, fStream);             // Fifth line represents the number of cases.
	*numCases = atoi(line);

	return 0;
}

/* Reads genotypic data from samples of a given kind (cases or controls). */
int readDatasetSamplesData(const char *fileName, uint *datasetSamplesPacked32, uint numSNPs, uint numSamplesPacked32) {

	size_t numElem;
	FILE *ifpSamples = fopen(fileName, "rb");

	uint numElemToRead = numSamplesPacked32 * numSNPs * SNP_CALC;

	numElem = fread(datasetSamplesPacked32, sizeof(unsigned int), numSamplesPacked32 * numSNPs * SNP_CALC, ifpSamples);

	if(numElem != numElemToRead) {
		std::cerr << "Problem loading samples from storage device" << std::endl;
		return 1;
	}

	fclose(ifpSamples);	
	return 0;
}

/* Calculates nCk (number of choices of 'k' items from 'n' items), i.e.  --->  n! / (k!(n-k)!)
   Used to calculate the achieved number of SNP combinations evaluated per second.
 */
unsigned long long n_choose_k(unsigned int n, unsigned int k)
{
	unsigned long long result = 1;		// nC0

	for (unsigned int i = 1; i <= k; i++) {	// nC1 until nCk
		result = result * n / i;	// calculates nC_{i} from nC_{i-1}
	n = n - 1;
	}

	return result;
}

/* Performs a fourth-order epistasis detection search on one of more GPUs.
 */
cudaError_t epistasisDetectionSearch(unsigned int* datasetCases_hostMatrixA, unsigned int* datasetControls_hostMatrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, int * roundsCounter, double * searchTime, float * outputScore, unsigned long long int * outputIndices) {

	cudaError_t result;

	/* Starts measuring time */
	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);       


	/* Scores and SNP combination indices for different GPUs. NUM_GPUS is set in the Makefile */
	float outputScore_omp_arr[NUM_GPUS];	
	unsigned long long int outputIndices_omp_arr[NUM_GPUS];


	/* Constructs lgamma() lookup table. Used for calculation of K2 Bayesian scores */
	int tablePrecalc_size = max(numCases, numControls) + 1;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	for(int i=1; i < (tablePrecalc_size + 1); i++) {
		h_tablePrecalc[i - 1] = lgamma((double)i);
	}


	(*roundsCounter) = 0;   

	double tensorTeraOperationsAcc = 0;


	/* Some parts of the code can be further simplified provided each MPI process is only assigned one GPU.
	 * This contrasts with the parallelization scheme that was being used before, which relied on OpenMP to facilitate targeting multiple GPUs on a single node. */
	
	// #pragma omp parallel num_threads(NUM_GPUS) reduction(+: tensorTeraOperationsAcc)
	// {

	int omp_thread_id = 0;	// omp_get_thread_num();
	cudaSetDevice(omp_thread_id);


	/* GPU memory allocation for cases */

	ScalarBinary32 *cases_A_ptrGPU;
	ScalarBinary32 *cases_B_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &cases_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC);		
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases input data." << std::endl;
	}


	result = cudaMemcpyAsync(cases_A_ptrGPU, datasetCases_hostMatrixA, sizeof(int) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);

	cases_B_ptrGPU = cases_A_ptrGPU;	/* Makes matrix B points to the same data as matrix A */

	int *C_ptrGPU_cases;
	result = cudaMalloc((int**) &C_ptrGPU_cases, sizeof(int) * NUM_STREAMS * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC));        

	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases output data." << std::endl;
	}


	/* GPU memory allocation for controls */

	ScalarBinary32 *controls_A_ptrGPU;
	ScalarBinary32 *controls_B_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &controls_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC);	
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls input data." << std::endl;
	}

	result = cudaMemcpyAsync(controls_A_ptrGPU, datasetControls_hostMatrixA, sizeof(int) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);

	controls_B_ptrGPU = controls_A_ptrGPU;	/* Makes matrix B points to the same data as matrix A */

	int *C_ptrGPU_controls;
	result = cudaMalloc((int**) &C_ptrGPU_controls, sizeof(int) * NUM_STREAMS * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC));          

	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls output data." << std::endl;
	}


	/* Copies K2 score lookup table to (each) GPU device */

	float * d_tablePrecalc;
	result = cudaMalloc((float**)&d_tablePrecalc, tablePrecalc_size * sizeof(float));
	result = cudaMemcpy(d_tablePrecalc, h_tablePrecalc, tablePrecalc_size * sizeof(float), cudaMemcpyHostToDevice);


	/* Allocates and initializes memory related to storing best score and indexes of corresponding set of SNPs */

	float * d_output;
	unsigned long long int * d_output_packedIndices;
	float h_output[1] = {FLT_MAX};
	result = cudaMalloc((float**)&d_output, 1 * sizeof(float));								
	result = cudaMalloc((unsigned long long int**)&d_output_packedIndices, 1 * sizeof(unsigned long long int));		
	result = cudaMemcpy(d_output, h_output, 1 * sizeof(float), cudaMemcpyHostToDevice);


	/* Setup of matrix-matrix operations using CUTLASS 2.X (tested on v2.5). */ 

	uint A_leadingDim_cases = numCasesWithPadding;             
	uint B_leadingDim_cases = numCasesWithPadding;             

	uint A_leadingDim_controls = numControlsWithPadding;       
	uint B_leadingDim_controls = numControlsWithPadding;       

	uint C_leadingDim = SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC;	


	/* Constructs contingency tables for individual SNPs */

	uint * d_output_individualSNP_popcountsForCases;
	uint * d_output_individualSNP_popcountsForControls;

	int blocksPerGrid_ind = (size_t)ceil(((float)(numSNPs)) / ((float)32));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls, 3 * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases, 3 * numSNPs * sizeof(uint));

	individualPopcount<<<blocksPerGrid_ind, 32, 0, 0>>>(0, (uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, numSNPs, numCases, numControls);  


	/* Constructs contingency tables for pairwise interactions of SNPs */

	uint * d_output_pairwiseSNP_popcountsForCases;
	uint * d_output_pairwiseSNP_popcountsForControls;
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForControls, 9 * numSNPs * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForCases, 9 * numSNPs * numSNPs * sizeof(uint));

	dim3 blocksPerGrid_pairwise ( (size_t)ceil(((float)(numSNPs)) / ((float)16)), (size_t)ceil(((float)(numSNPs)) / ((float)16)), 1 );
	dim3 workgroupSize_pairwise ( 16, 16, 1 );

	pairwisePopcount<<<blocksPerGrid_pairwise, workgroupSize_pairwise, 0, 0>>>((uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls, 0);


	/* Allocates space for pairwise combination of SNPs in CUDA cores.
	   Y_Z uses more space (x NUM_STREAMS) if using multiple streams for enabling concurrent execution of inner-loop (Z) iterations. */

	uint * d_output_WX_cases;
	uint * d_output_WX_controls;
	result = cudaMalloc((uint**)&d_output_WX_cases, (numCasesWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));		
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases pairwise popcounts." << std::endl;
	}
	result = cudaMalloc((uint**)&d_output_WX_controls, (numControlsWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));	
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls pairwise popcounts." << std::endl;
	}

	uint * d_output_YZ_cases;
	uint * d_output_YZ_controls;
	result = cudaMalloc((uint**)&d_output_YZ_cases, NUM_STREAMS * (numCasesWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases pairwise popcounts." << std::endl;
	}
	result = cudaMalloc((uint**)&d_output_YZ_controls, NUM_STREAMS * (numControlsWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls pairwise popcounts." << std::endl;
	}

	uint * d_output_XY_cases;
	uint * d_output_XY_controls;
	result = cudaMalloc((uint**)&d_output_XY_cases, (numCasesWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases pairwise popcounts." << std::endl;
	}
	result = cudaMalloc((uint**)&d_output_XY_controls, (numControlsWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls pairwise popcounts." << std::endl;
	}

	uint * d_output_WY_cases;
	uint * d_output_WY_controls;
	result = cudaMalloc((uint**)&d_output_WY_cases, (numCasesWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases pairwise popcounts." << std::endl;
	}
	result = cudaMalloc((uint**)&d_output_WY_controls, (numControlsWithPadding / 32) * (SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK) * sizeof(uint));
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls pairwise popcounts." << std::endl;
	}



	/* Allocate memory for 3-way popcounts (calculated on tensor cores) */

	int *C_ptrGPU_cases_XYZ;
	result = cudaMalloc((int**) &C_ptrGPU_cases_XYZ, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));
	int *C_ptrGPU_controls_XYZ;
	result = cudaMalloc((int**) &C_ptrGPU_controls_XYZ, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));

	int *C_ptrGPU_cases_WYZ;
	result = cudaMalloc((int**) &C_ptrGPU_cases_WYZ, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));
	int *C_ptrGPU_controls_WYZ;
	result = cudaMalloc((int**) &C_ptrGPU_controls_WYZ, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));

	int *C_ptrGPU_cases_WXY;
	result = cudaMalloc((int**) &C_ptrGPU_cases_WXY, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));
	int *C_ptrGPU_controls_WXY;
	result = cudaMalloc((int**) &C_ptrGPU_controls_WXY, sizeof(int) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (numSNPs * SNP_CALC));



	/* CUDA stream creation */

	cudaStream_t cudaStream_prework_k3_WX, cudaStream_prework_k3_XY, cudaStream_prework_k3_WY;	
	cudaStreamCreate(&cudaStream_prework_k3_WX);
	cudaStreamCreate(&cudaStream_prework_k3_XY);
	cudaStreamCreate(&cudaStream_prework_k3_WY);

	cudaStream_t cudaStreamToUse[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&cudaStreamToUse[i]);
	}


	/* Main loop performing the SNP evaluation rounds. */

	MPI_Status stat;
	int start_W;
	int start_X;

	unsigned long long start_W_and_X;

	uint objectiveFunctionIndex = 0;

	// #pragma omp for schedule(TYPE_SCHEDULING)
	// for(int start_W = 0; start_W < numSNPsWithPadding; start_W+=SNP_BLOCK) {
	while (1) {

		MPI_Send (NULL, 0 , MPI_INT, 0 /* goes to rank 0 */, 0 /* only one type of tag */, MPI_COMM_WORLD);	

		// Gets data from master
		MPI_Recv (&start_W_and_X , 1, MPI_UNSIGNED_LONG_LONG, 0 /* comes from rank 0 */, 0 /* only one type of tag */, MPI_COMM_WORLD, &stat);

		if (start_W_and_X == 0xffffffffffffffff) {	
			break;
		}

		start_W = start_W_and_X & 0xFFFFFFFF;
		start_X = (start_W_and_X >> 32) & 0xFFFFFFFF; 

		std::cout << "Loop iteration W=" << (int) (start_W / SNP_BLOCK) << " and X=" << (int) (start_X / SNP_BLOCK) << " out of " << (int) (numSNPsWithPadding / SNP_BLOCK) << std::endl;	

		/* In case the last calls to the 'applyScore_and_FindGloballyBestSol()' GPU kernel did not terminate yet at this point. */
		for(int i=0; i<NUM_STREAMS; i++) {
			cudaStreamSynchronize(cudaStreamToUse[i]);
		}


		/* Combines an SNP W with a block of SNPs X. */

		dim3 blocksPerGrid_prework_k3_WX( (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), 1);
		dim3 workgroupSize_prework_k3_WX( 1, 1, 64 );     
		combine<<<blocksPerGrid_prework_k3_WX, workgroupSize_prework_k3_WX, 0, cudaStream_prework_k3_WX >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), d_output_WX_cases, d_output_WX_controls, numSNPs, numCases, numControls, start_W, start_X); 

		/* ** TENSOR ** Constructs contingency tables for W_X_Y (also used as W_X_Z) */

		// Processes Cases
		int SNP_Y_index_start = ((int)(start_X / 64.0)) * 64;	
		ScalarBinary32 *A_ptrGPU_iter_cases_WXY = (ScalarBinary32 *) (d_output_WX_cases);   
		ScalarBinary32 *B_ptrGPU_iter_cases_WXY = cases_B_ptrGPU  +  (SNP_Y_index_start * SNP_CALC * (numCasesWithPadding/32));	


		result = cutlass_U1_MmaMatOpTN(
				(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
				(numSNPsWithPadding - SNP_Y_index_start) * (SNP_CALC),
				numCasesWithPadding,
				(cutlass::uint1b_t*) A_ptrGPU_iter_cases_WXY,
				A_leadingDim_cases,
				(cutlass::uint1b_t*) B_ptrGPU_iter_cases_WXY,
				B_leadingDim_cases,
				C_ptrGPU_cases_WXY + (SNP_Y_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),	
				C_leadingDim,   
				cudaStream_prework_k3_WX 
				);

		if(result != cudaSuccess) {
			std::cout << "Problem in construction of contingency tables for W_X_Y (cases)." << std::endl;
		}

		tensorTeraOperationsAcc += (double) numCasesWithPadding * (double)(SNP_BLOCK * SNP_BLOCK) * (double)(SNP_CALC * SNP_CALC) * (double)(numSNPsWithPadding - SNP_Y_index_start) * (double)(SNP_CALC);


		// Processes Controls

		ScalarBinary32 *A_ptrGPU_iter_controls_WXY = (ScalarBinary32 *) (d_output_WX_controls);     
		ScalarBinary32 *B_ptrGPU_iter_controls_WXY = controls_B_ptrGPU  +  (SNP_Y_index_start * SNP_CALC * (numControlsWithPadding/32));

		result = cutlass_U1_MmaMatOpTN(
				(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
				(numSNPsWithPadding - SNP_Y_index_start) * (SNP_CALC),
				numControlsWithPadding,
				(cutlass::uint1b_t*)A_ptrGPU_iter_controls_WXY,
				A_leadingDim_controls,
				(cutlass::uint1b_t*) B_ptrGPU_iter_controls_WXY,
				B_leadingDim_controls,
				C_ptrGPU_controls_WXY	+       (SNP_Y_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),
				C_leadingDim,   
				cudaStream_prework_k3_WX	
				);

		if(result != cudaSuccess) {
			std::cout << "Problem in construction of contingency tables for W_X_Y (controls)." << std::endl;
		}

		tensorTeraOperationsAcc += (double) numControlsWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (numSNPsWithPadding - SNP_Y_index_start) * (double) (SNP_CALC);



		for(int start_Y = start_X; start_Y < numSNPsWithPadding; start_Y+=SNP_BLOCK) {

			/* In case the last calls to the 'applyScore_and_FindGloballyBestSol()' GPU kernel did not terminate yet at this point. */
			for(int i=0; i<NUM_STREAMS; i++) {
				cudaStreamSynchronize(cudaStreamToUse[i]);
			}


			/* Combines a block of SNPs X with a block of SNPs Y. Used for the construction of contingency tables for 3rd order SNP interactions. */

			dim3 blocksPerGrid_prework_k3_XY( (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), 1);
			dim3 workgroupSize_prework_k3_XY( 1, 1, 64 );     
			combine<<<blocksPerGrid_prework_k3_XY, workgroupSize_prework_k3_XY, 0, cudaStream_prework_k3_XY >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), d_output_XY_cases, d_output_XY_controls, numSNPs, numCases, numControls, start_X, start_Y);


			/* Combines a block of SNPs W with a block of SNPs Y. Used for the construction of contingency tables for 3rd order SNP interactions. */

			dim3 blocksPerGrid_prework_k3_WY( (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), 1);
			dim3 workgroupSize_prework_k3_WY( 1, 1, 64 );     
			combine<<<blocksPerGrid_prework_k3_WY, workgroupSize_prework_k3_WY, 0, cudaStream_prework_k3_WY >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), d_output_WY_cases, d_output_WY_controls, numSNPs, numCases, numControls, start_W, start_Y);	


			/* ** TENSOR ** Constructs contingency tables for W_Y_Z */

			// Processes Cases

			int SNP_Z_index_start = ((int)(start_Y / 64.0)) * 64;	
			ScalarBinary32 *A_ptrGPU_iter_cases_WYZ = (ScalarBinary32 *) (d_output_WY_cases);   
			ScalarBinary32 *B_ptrGPU_iter_cases_WYZ = cases_B_ptrGPU +  (SNP_Z_index_start * SNP_CALC * (numCasesWithPadding/32));        

			result = cutlass_U1_MmaMatOpTN(
					(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
					(numSNPsWithPadding - SNP_Z_index_start) * (SNP_CALC),
					numCasesWithPadding,
					(cutlass::uint1b_t*) A_ptrGPU_iter_cases_WYZ,
					A_leadingDim_cases,
					(cutlass::uint1b_t*) B_ptrGPU_iter_cases_WYZ,
					B_leadingDim_cases,
					C_ptrGPU_cases_WYZ +	(SNP_Z_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),
					C_leadingDim,   
					cudaStream_prework_k3_WY	
					);

			if(result != cudaSuccess) {
				std::cout << "Problem in construction of contingency tables for W_Y_Z (cases)." << std::endl;
			}


			tensorTeraOperationsAcc += (double) numCasesWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (numSNPsWithPadding - SNP_Z_index_start) * (double) (SNP_CALC);


			// Processes Controls

			ScalarBinary32 *A_ptrGPU_iter_controls_WYZ = (ScalarBinary32 *) (d_output_WY_controls);     
			ScalarBinary32 *B_ptrGPU_iter_controls_WYZ = controls_B_ptrGPU +  (SNP_Z_index_start * SNP_CALC * (numControlsWithPadding/32));

			result = cutlass_U1_MmaMatOpTN(
					(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
					(numSNPsWithPadding - SNP_Z_index_start) * (SNP_CALC),
					numControlsWithPadding,
					(cutlass::uint1b_t*)A_ptrGPU_iter_controls_WYZ,
					A_leadingDim_controls,
					(cutlass::uint1b_t*) B_ptrGPU_iter_controls_WYZ,
					B_leadingDim_controls,
					C_ptrGPU_controls_WYZ +	(SNP_Z_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),
					C_leadingDim,   
					cudaStream_prework_k3_WY	
					);

			if(result != cudaSuccess) {
				std::cout << "Problem in construction of contingency tables for W_Y_Z (controls)." << std::endl;
			}

			tensorTeraOperationsAcc += (double) numControlsWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (numSNPsWithPadding - SNP_Z_index_start) * (double) (SNP_CALC);


			/* ** TENSOR ** Constructs contingency tables for X_Y_Z */

			// Processes Cases

			ScalarBinary32 *A_ptrGPU_iter_cases_XYZ = (ScalarBinary32 *) (d_output_XY_cases);   
			ScalarBinary32 *B_ptrGPU_iter_cases_XYZ = cases_B_ptrGPU +  (SNP_Z_index_start * SNP_CALC * (numCasesWithPadding/32));        

			result = cutlass_U1_MmaMatOpTN(
					(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
					(numSNPsWithPadding - SNP_Z_index_start) * (SNP_CALC),
					numCasesWithPadding,
					(cutlass::uint1b_t*) A_ptrGPU_iter_cases_XYZ,
					A_leadingDim_cases,
					(cutlass::uint1b_t*) B_ptrGPU_iter_cases_XYZ,
					B_leadingDim_cases,
					C_ptrGPU_cases_XYZ +       (SNP_Z_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),
					C_leadingDim,   
					cudaStream_prework_k3_XY	
					);

			if(result != cudaSuccess) {
				std::cout << "Problem in construction of contingency tables for X_Y_Z (cases)." << std::endl;
			}


			tensorTeraOperationsAcc += (double) numCasesWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (numSNPsWithPadding - SNP_Z_index_start) * (double) (SNP_CALC);


			// Processes Controls

			ScalarBinary32 *A_ptrGPU_iter_controls_XYZ = (ScalarBinary32 *) (d_output_XY_controls);     
			ScalarBinary32 *B_ptrGPU_iter_controls_XYZ = controls_B_ptrGPU +  (SNP_Z_index_start * SNP_CALC * (numControlsWithPadding/32));

			result = cutlass_U1_MmaMatOpTN(
					(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
					(numSNPsWithPadding - SNP_Z_index_start) * (SNP_CALC),
					numControlsWithPadding,
					(cutlass::uint1b_t*)A_ptrGPU_iter_controls_XYZ,
					A_leadingDim_controls,
					(cutlass::uint1b_t*) B_ptrGPU_iter_controls_XYZ,
					B_leadingDim_controls,
					C_ptrGPU_controls_XYZ +	(SNP_Z_index_start * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC * SNP_CALC))),
					C_leadingDim,   
					cudaStream_prework_k3_XY 
					);

			if(result != cudaSuccess) {
				std::cout << "Problem in construction of contingency tables for X_Y_Z (controls)." << std::endl;
			}

			tensorTeraOperationsAcc += (double) numControlsWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (numSNPsWithPadding - SNP_Z_index_start) * (double) (SNP_CALC);


			for(int start_Z = start_Y; start_Z < numSNPsWithPadding; start_Z+=SNP_BLOCK) {

				// #pragma omp atomic	
				(*roundsCounter)++;


				/* Combines a block of SNPs Y with a block of SNPs Z. */

				dim3 blocksPerGrid_prework_k3_YZ( (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), 1);
				dim3 workgroupSize_prework_k3_YZ( 1, 1, 64 );     
				combine<<<blocksPerGrid_prework_k3_YZ, workgroupSize_prework_k3_YZ, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), d_output_YZ_cases + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC) * (numCasesWithPadding / 32)), d_output_YZ_controls + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC) * (numControlsWithPadding / 32)), numSNPs, numCases, numControls, start_Y, start_Z);     


				if((start_Y == start_X) && (start_Z == start_Y) ) {	
					cudaStreamSynchronize(cudaStream_prework_k3_WX);	
				}



				/* Main calculation in 4-way (using tensor cores). 
				   Combines block of (SNP_BLOCK * SNP_BLOCK) 2-way pairings (WX) with (SNP_BLOCK * SNP_BLOCK) 2-way pairings (YZ) of SNPs.
				 */


				// Processes Cases

				ScalarBinary32 *A_ptrGPU_iter_cases = (ScalarBinary32 *) (d_output_WX_cases);	
				ScalarBinary32 *B_ptrGPU_iter_cases = (ScalarBinary32 *) (d_output_YZ_cases) + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC) * (numCasesWithPadding / 32)); 

				result = cutlass_U1_MmaMatOpTN(
						(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),   
						(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),       
						numCasesWithPadding,            
						(cutlass::uint1b_t*) A_ptrGPU_iter_cases,
						A_leadingDim_cases,
						(cutlass::uint1b_t*) B_ptrGPU_iter_cases,
						B_leadingDim_cases,
						C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC)) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC))),  
						C_leadingDim,
						cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] 
						);

				if(result != cudaSuccess) {
					std::cout << "Problem in construction of contingency tables for W_X_Y_Z (cases)." << std::endl;
				}

				tensorTeraOperationsAcc += (double) numCasesWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC);


				// Processes Controls

				ScalarBinary32 *A_ptrGPU_iter_controls = (ScalarBinary32 *) (d_output_WX_controls);	
				ScalarBinary32 *B_ptrGPU_iter_controls = (ScalarBinary32 *) (d_output_YZ_controls) + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC) * (numControlsWithPadding / 32));

				result = cutlass_U1_MmaMatOpTN(
						(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
						(SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC),
						numControlsWithPadding,         
						(cutlass::uint1b_t*)A_ptrGPU_iter_controls,
						A_leadingDim_controls,
						(cutlass::uint1b_t*) B_ptrGPU_iter_controls,
						B_leadingDim_controls,
						C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC)) * ((SNP_BLOCK * SNP_BLOCK) * (SNP_CALC * SNP_CALC))),
						C_leadingDim,
						cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] 
						);

				if(result != cudaSuccess) {
					std::cout << "Problem in construction of contingency tables for W_X_Y_Z (controls)." << std::endl;
				}

				tensorTeraOperationsAcc += numControlsWithPadding * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC) * (double) (SNP_BLOCK * SNP_BLOCK) * (double) (SNP_CALC * SNP_CALC);


				if(start_Z == start_Y) {
					cudaStreamSynchronize(cudaStream_prework_k3_WY);    
					cudaStreamSynchronize(cudaStream_prework_k3_XY);
				}


				/* Call K2 objective scoring function */

				dim3 blocksPerGrid_objFun( (size_t)ceil(((float)(SNP_BLOCK)) / ((float)SNP_BLOCK) ), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)), (size_t)ceil(((float)(SNP_BLOCK)) / ((float)1)));		
				dim3 workgroupSize_objFun( SNP_BLOCK, 1, 1 );	

				applyScore_and_FindGloballyBestSol<<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC)), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * ((SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC) * (SNP_BLOCK * SNP_BLOCK * SNP_CALC * SNP_CALC)), C_ptrGPU_cases_XYZ, C_ptrGPU_controls_XYZ, C_ptrGPU_cases_WYZ, C_ptrGPU_controls_WYZ, C_ptrGPU_cases_WXY, C_ptrGPU_controls_WXY, d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_packedIndices, start_W, start_X, start_Y, start_Z, numSNPs, numCases, numControls);


				objectiveFunctionIndex = (objectiveFunctionIndex + 1) % NUM_STREAMS;	

			}
		}
	}



	/* In case evaluation rounds are still executing */
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamSynchronize(cudaStreamToUse[i]);
	}

	/* Copies best solution found from GPU memory to Host */
	cudaMemcpy(&outputScore_omp_arr[omp_thread_id], d_output, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&outputIndices_omp_arr[omp_thread_id], d_output_packedIndices, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(cases_A_ptrGPU);
	cudaFree(C_ptrGPU_cases);
	cudaFree(controls_A_ptrGPU);
	cudaFree(C_ptrGPU_controls);
	cudaFree(d_output_individualSNP_popcountsForControls);
	cudaFree(d_output_individualSNP_popcountsForCases);
	cudaFree(d_tablePrecalc);
	cudaFree(d_output);
	cudaFree(d_output_packedIndices);

	// }	// closes 'pragma omp parallel'


	*outputScore = FLT_MAX;
	for(int i=0; i<NUM_GPUS; i++) {
		if(outputScore_omp_arr[i] < (*outputScore)) {
			*outputScore = outputScore_omp_arr[i];
			*outputIndices = outputIndices_omp_arr[i];
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &t_end); // final timestamp

	(*searchTime) = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

	std::cout << "Tensor TOPS: " << std::fixed << std::setprecision(3) << ((double) (tensorTeraOperationsAcc / (*searchTime) * 2) / 1e12) << std::endl;	// '* 2' because each AND+POC (or XOR+POPC) counts as two operations

	free(h_tablePrecalc);

	return cudaSuccess;
}


/* Entry point of the application. */
int main(int argc, const char *arg[]) {

	cudaError_t result;

	uint numSNPs, numControls, numCases;
	char controlsFileName[MAX_CHAR_ARRAY], casesFileName[MAX_CHAR_ARRAY];

	if(argc < 2) {
		std::cerr << "Usage: epi4tensor dataset.txt" << std::endl;
		return 1;
	}

	char *ts = strdup(arg[1]);
	char *pathToDataset = dirname(ts);

	/* Reads information about input dataset (number of SNPs, controls and cases, and controls/cases file names) from description file. */
	readDatasetDescriptionFile(arg[1], &numSNPs, controlsFileName, &numControls, casesFileName, &numCases);

	/* Calculates number of distinct blocks and padds number of SNPs to process to the block size. */
	uint numBlocks = ceil((float)numSNPs / (float)SNP_BLOCK);
	uint numSNPsWithPadding = numBlocks * SNP_BLOCK;

	/* Padds the number of controls and of cases. */
	uint numCasesWithPadding = ceil((float)numCases / PADDING_SAMPLES) * PADDING_SAMPLES;	
	uint numControlsWithPadding = ceil((float)numControls / PADDING_SAMPLES) * PADDING_SAMPLES;


	/* Prints information about dataset and number of distinct blocks of SNPs to process. */
	std::cout << "Num. SNPs: " << numSNPs << std::endl;
	std::cout << "Num. Cases: " << numCases << std::endl;
	std::cout << "Num. Controls: " << numControls << std::endl;


	/* Allocates pinned memory for holding controls and cases dataset matrices.
	   Each 32-bit 'unsigned int' holds 32 binary values representing genotype information.
	   Only two allele types are represented (SNP_CALC macro equals 2), ...
	   ... being information about the third allele type infered.
	 */


	int numCasesPacked32 = ceil(((float) numCasesWithPadding) / 32.0f);
	int numControlsPacked32 = ceil(((float) numControlsWithPadding) / 32.0f);

	int datasetCasesPacked32_size = numCasesPacked32 * numSNPsWithPadding * SNP_CALC;
	int datasetControlsPacked32_size = numControlsPacked32 * numSNPsWithPadding * SNP_CALC;

	unsigned int *datasetCasesPacked32 = NULL, *datasetControlsPacked32 = NULL;

	result = cudaHostAlloc((void**)&datasetCasesPacked32, datasetCasesPacked32_size * sizeof(unsigned int), cudaHostAllocDefault );     
	result = cudaHostAlloc((void**)&datasetControlsPacked32, datasetControlsPacked32_size * sizeof(unsigned int), cudaHostAllocDefault );
	if((datasetCasesPacked32 == NULL) || (datasetControlsPacked32 == NULL)) {
		std::cerr << "Problem allocating Host memory for cases and/or controls" << std::endl;
	}


	/* Reads dataset (controls and cases data) from storage device.
	   Input dataset must be padded with zeros in the dimension of samples (cases / controls), ...
	   ... making the number of bits per {SNP, allele} tuple a multiple of PADDING_SAMPLES. */

	std::string absolutePathToCasesFile = std::string(pathToDataset) + "/" + casesFileName;
	std::string absolutePathToControlsFile = std::string(pathToDataset) + "/" + controlsFileName;

	readDatasetSamplesData(absolutePathToCasesFile.c_str(), datasetCasesPacked32, numSNPs, numCasesPacked32);
	readDatasetSamplesData(absolutePathToControlsFile.c_str(), datasetControlsPacked32, numSNPs, numControlsPacked32);


	std::cout << "-------------------------------" << std::endl;




	/* Initializes the MPI environment */
	MPI_Init(NULL, NULL);

	/* Gets the number of processes */
	int mpi_world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

	/* Gets the rank of the process */
	int mpi_world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);


	/* Gets the name of the processor */
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	/* Prints a hello world message */
	printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, mpi_world_rank, mpi_world_size);

	if(mpi_world_rank == 0) {
		printf("Using %d MPI processes.\n", mpi_world_size);
	}

	MPI_Barrier(MPI_COMM_WORLD); /* All nodes are ready */

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp


	unsigned int start_W = 0;
	unsigned int start_X = 0;	

	unsigned long long start_W_and_X;

	int numActiveSlaves = mpi_world_size - 1;	

	float outputScore = FLT_MAX;
	unsigned long long int outputIndices;


	if (mpi_world_rank == 0) {

		MPI_Status mpiStatus;

		while ( numActiveSlaves > 0 ) {

			MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0 /* only one type of tag */, MPI_COMM_WORLD, &mpiStatus);
			int slaveRank = mpiStatus.MPI_SOURCE;


			if (start_W < numSNPsWithPadding) {	

				start_W_and_X = (((unsigned long long int) start_W) << 0) | (((unsigned long long int) start_X) << 32);
				MPI_Send(&start_W_and_X, 1, MPI_UNSIGNED_LONG_LONG , slaveRank, 0 /* only one type of tag */, MPI_COMM_WORLD);

				start_X += SNP_BLOCK;

				if(start_X >= numSNPsWithPadding) {	
					start_W += SNP_BLOCK;
					start_X = start_W;
				}



			} else {

				unsigned long long stopSlave = 0xffffffffffffffff;	
				MPI_Send (&stopSlave, 1, MPI_UNSIGNED_LONG_LONG, slaveRank , 0 /* only one type of tag */ , MPI_COMM_WORLD);

				numActiveSlaves--;
			}
		}


	} else {

		/* Launches epistasis detection search. */

		int roundsCounter;
		double searchTime;

		result = epistasisDetectionSearch(
				datasetCasesPacked32,			// Cases matrix.
				datasetControlsPacked32,		// Controls matrix.
				numSNPs,                           	// Number of SNPs.
				numCases,                           	// Number of cases.
				numControls,                        	// Number of controls.
				numSNPsWithPadding,                 	// Number of SNPs padded to block size.
				numCasesWithPadding,     		// Number of cases padded to PADDING_SIZE.
				numControlsWithPadding,     		// Number of controls padded to PADDING_SIZE.
				&roundsCounter,			// Counter for number of rounds processed.
				&searchTime,				// Counter for execution time (seconds).
				&outputScore,				// Score of best score found.
				&outputIndices				// Indices of SNPs of set that results in best score.
				);

		if(result != 0) {
			std::cerr << "Epistasis detection search failed." << std::endl;
		}

	}


	/* Reduces scores and gets indices of best score */

	float * outputScorePerMpiProcess; 
	unsigned long long int * outputIndicesPerMpiProcess;

	if ( mpi_world_rank == 0) { 
		outputScorePerMpiProcess = (float *)malloc(mpi_world_size*1*sizeof(float)); 
		outputIndicesPerMpiProcess = (unsigned long long int *)malloc(mpi_world_size*1*sizeof(unsigned long long int));

	} 

	MPI_Gather( &outputScore, 1, MPI_FLOAT, outputScorePerMpiProcess, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);	// 0 is the root process
	MPI_Gather( &outputIndices, 1, MPI_UNSIGNED_LONG_LONG, outputIndicesPerMpiProcess, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

	float outputScoreAllMpiProcesses = FLT_MAX;
	unsigned long long outputIndicesAllMpiProcesses;

	if(mpi_world_rank == 0) {
		for(int i = 0; i < mpi_world_size; i++) {
			if(outputScorePerMpiProcess[i] < outputScoreAllMpiProcesses) {
				outputScoreAllMpiProcesses = outputScorePerMpiProcess[i];
				outputIndicesAllMpiProcesses = outputIndicesPerMpiProcess[i];
			}
		}

	}


	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_MONOTONIC, &t_end);

	MPI_Finalize();


	if(mpi_world_rank == 0) {

		/* Prints indices of set of SNPs most associated with phenotype and its score */

		double timing_duration_mpi = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

		std::cout << "-------------------------------" << std::endl << "{SNP_W_i, SNP_X_i, SNP_Y_i, SNP_Z_i}: SCORE\t->\t{" << ((outputIndicesAllMpiProcesses >> 0) & 0xFFFF) << ", " << ((outputIndicesAllMpiProcesses >> 16) & 0xFFFF) << ", " << ((outputIndicesAllMpiProcesses >> 32) & 0xFFFF) << ", " << ((outputIndicesAllMpiProcesses >> 48) & 0xFFFF) << "}: " << std::fixed << std::setprecision(6) << outputScoreAllMpiProcesses << std::endl;

		/* Prints time to execute the application, the achieved performance, and the ratio of unique sets */

		unsigned long long numCombinations = n_choose_k(numSNPs, INTER_OR);

		std::cout << "Wall-clock time:\t" << std::fixed << std::setprecision(3) << timing_duration_mpi << " seconds" << std::endl;    

		std::cout << "Num. unique sets per sec. (scaled to sample size): " << std::fixed << std::setprecision(3) << (((double) numCombinations * (double) (numCases + numControls) / (double)(timing_duration_mpi)) / 1e12) << " × 10^12" << std::endl;

		std::cout << "Unique sets of SNPs evaluated (k=" << INTER_OR << "): " << numCombinations << std::endl;

	}
	return result == cudaSuccess ? 0 : 1;	

}





