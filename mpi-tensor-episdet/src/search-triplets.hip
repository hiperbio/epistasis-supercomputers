#include "hip/hip_runtime.h"
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <hipblas.h>
#include <rocblas.h>
#include <mpi.h>
#include "helper_amd.hpp"


hipError_t EpistasisDetectionSearch(unsigned int* datasetCases_host_matrixA, unsigned int* datasetControls_host_matrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, int * roundsCounter, double * searchTime, float * outputFromGpu, unsigned long long int * output_indexFromGpu_packedIndices) {
	hipError_t result;

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp


        /* Allocate GPU memory for Cases */

	ScalarBinary32 *cases_A_ptrGPU;
	ScalarBinary32 *cases_B_ptrGPU;
	result = hipMalloc((ScalarBinary32 **) &cases_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC);		
        if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for cases input data." << std::endl;
        }
	

	result = hipMemcpyAsync(cases_A_ptrGPU, datasetCases_host_matrixA, sizeof(int) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC, hipMemcpyHostToDevice, 0);

	cases_B_ptrGPU = cases_A_ptrGPU;	// Matrix B points to the same data as matrix A

	float *C_ptrGPU_cases;
	result = hipMalloc((float**) &C_ptrGPU_cases, sizeof(float) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));        

	if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for cases output data." << std::endl;
	}


        /* Allocate GPU memory for Controls */

	ScalarBinary32 *controls_A_ptrGPU;
	ScalarBinary32 *controls_B_ptrGPU;
	result = hipMalloc((ScalarBinary32 **) &controls_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC);	
	if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for controls input data." << std::endl;
	}

	result = hipMemcpyAsync(controls_A_ptrGPU, datasetControls_host_matrixA, sizeof(int) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC, hipMemcpyHostToDevice, 0);

	controls_B_ptrGPU = controls_A_ptrGPU;	// Matrix B points to the same data as matrix A

	float *C_ptrGPU_controls;
	result = hipMalloc((float**) &C_ptrGPU_controls, sizeof(float) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));          

	if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for controls output data." << std::endl;
	}


	/* Allocates blocks for representing 16-bit values.
	   Stores "NUM_STREAMS * blocks", for each sample type (case or control).
	 */

        __half * cases_A_block16bit_ptrGPU;
        __half * controls_A_block16bit_ptrGPU;
        result = hipMalloc((__half**)&cases_A_block16bit_ptrGPU, numCasesWithPadding * (SNP_CALC * SNP_CALC) * BLOCK_SIZE * sizeof(__half));
        if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for cases 16-bit data matrix A." << std::endl;
        }
        result = hipMalloc((__half**)&controls_A_block16bit_ptrGPU, numControlsWithPadding * (SNP_CALC * SNP_CALC) * BLOCK_SIZE * sizeof(__half));
        if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for controls 16-bit data matrix A." << std::endl;
        }


	__half * cases_B_block16bit_ptrGPU;
        __half * controls_B_block16bit_ptrGPU;
        result = hipMalloc((__half**)&cases_B_block16bit_ptrGPU, NUM_STREAMS * numCasesWithPadding * (SNP_CALC) * BLOCK_SIZE * sizeof(__half));
        if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for cases 16-bit data matrix B.." << std::endl;
        }
        result = hipMalloc((__half**)&controls_B_block16bit_ptrGPU, NUM_STREAMS * numControlsWithPadding * (SNP_CALC) * BLOCK_SIZE * sizeof(__half));
        if(result != hipSuccess) {
                std::cerr << "Failed allocating memory for controls 16-bit data matrix B." << std::endl;
        }



	/* Computes lookup table on the Host */

	float * d_tablePrecalc;

	#if defined(MI_SCORE)
	// Mutual Information scoring
        int tablePrecalc_size = numCases + numControls;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	float numPatientsInv = 1.0 / (numCases + numControls);    
	h_tablePrecalc[0] = 0;
	for(int i=1; i < tablePrecalc_size; i++) {
		h_tablePrecalc[i] = log2((double)i * numPatientsInv);
	}
	#else 
	// K2 Bayesian scoring
        int tablePrecalc_size = max(numCases, numControls) + 1;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	for(int i=1; i < (tablePrecalc_size + 1); i++) {
		h_tablePrecalc[i - 1] = lgamma((double)i);
	}
	#endif

        result = hipMalloc((float**)&d_tablePrecalc, tablePrecalc_size * sizeof(float));
	result = hipMemcpy(d_tablePrecalc, h_tablePrecalc, tablePrecalc_size * sizeof(float), hipMemcpyHostToDevice);


	float * d_output;
	unsigned long long int * d_output_packedIndices;

	#if defined(MI_SCORE)
        float h_output[1] = {FLT_MIN};
	#else
	float h_output[1] = {FLT_MAX};
	#endif

        /* Stores the best score and indexes of corresponding set of SNPs */
	result = hipMalloc((float**)&d_output, 1 * sizeof(float));								
	result = hipMalloc((unsigned long long int**)&d_output_packedIndices, 1 * sizeof(unsigned long long int));		

	result = hipMemcpy(d_output, h_output, 1 * sizeof(float), hipMemcpyHostToDevice);


	uint A_leadingDim_cases = numCasesWithPadding;	  		
	uint B_leadingDim_cases = numCasesWithPadding;	  		

	uint A_leadingDim_controls = numControlsWithPadding;	  	
	uint B_leadingDim_controls = numControlsWithPadding;	  	

	uint C_leadingDim = BLOCK_SIZE * SNP_CALC * SNP_CALC;	


	/* Calculates individual population counts */

	uint * d_output_individualSNP_popcountsForCases;
	uint * d_output_individualSNP_popcountsForControls;

	
	(*roundsCounter) = 0;	

	/* Calculates pairwise population counts */

	/* For storing the popcounts of all SNP pairs. */
	uint * d_output_pairwiseSNP_popcountsForCases;
	uint * d_output_pairwiseSNP_popcountsForControls;
	result = hipMalloc((uint**)&d_output_pairwiseSNP_popcountsForControls, 9 * BLOCK_SIZE * numSNPs * sizeof(uint));
	result = hipMalloc((uint**)&d_output_pairwiseSNP_popcountsForCases, 9 * BLOCK_SIZE * numSNPs * sizeof(uint));


	/* For storing the popcounts of all SNP pairs. */
	uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases;
	uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls;
	result = hipMalloc((uint**)&d_output_pairwiseSNP_singleX_Z_popcountsForControls, 9 * 1 * numSNPs * sizeof(uint));
	result = hipMalloc((uint**)&d_output_pairwiseSNP_singleX_Z_popcountsForCases, 9 * 1 * numSNPs * sizeof(uint));


	/* HIP stream creation */

	hipStream_t hipStreamPairwiseSNPs, hipStream_singleX_Z;
	hipStreamCreate(&hipStreamPairwiseSNPs);
	hipStreamCreate(&hipStream_singleX_Z);

	hipStream_t hipStreamToUse[NUM_STREAMS];
	hipblasHandle_t hipblasHandlers[NUM_STREAMS];

	// Create hipBLAS handlers and bind to streams (1 to 1)
	for (int i = 0; i < NUM_STREAMS; i++) {
        	hipblasCreate(&hipblasHandlers[i]);
		hipStreamCreate(&hipStreamToUse[i]);
		hipblasSetStream(hipblasHandlers[i], hipStreamToUse[i]);
	}

        hipblasOperation_t transa = HIPBLAS_OP_T;
        hipblasOperation_t transb = HIPBLAS_OP_N;

	const float alpha = 1.0;
	const float beta = 0.0;

        uint objectiveFunctionIndex = 0;

	MPI_Status stat;
	int start_Y;

	while (1) {

		MPI_Send (NULL, 0 , MPI_INT, 0 /* goes to rank 0 */, 0 /* only one type of tag */, MPI_COMM_WORLD);

		// Gets data from master
		MPI_Recv (&start_Y , 1, MPI_INT, 0 /* comes from rank 0 */, 0 /* only one type of tag */, MPI_COMM_WORLD, &stat);

		if (start_Y == -1) {	// '-1' means there is no more data to process
			break;
		}


                std::cout << "Outer loop iteration " << (int) (start_Y / BLOCK_SIZE) << " out of " << (int) (numSNPsWithPadding / BLOCK_SIZE) << std::endl;

		/* In case the last calls to the objective function kernel did not terminate yet at this point. */
		for(int i=0; i<NUM_STREAMS; i++) {
			hipStreamSynchronize(hipStreamToUse[i]);
		}

		/* Calculation of pairwise population counts as part of the 3-way search process. */

		dim3 blocksPerGrid_pairwise ( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)16)), (size_t)ceil(((float)(numSNPs)) / ((float)16)), 1 );
		dim3 workgroupSize_pairwise ( 16, 16, 1 );   

		hipLaunchKernelGGL(epistasis_pairwiseSNPs, blocksPerGrid_pairwise, workgroupSize_pairwise, 0, hipStreamPairwiseSNPs, (uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls, start_Y);  


		for(int X_index=0; (X_index < (start_Y + BLOCK_SIZE)) && (X_index < numSNPs); X_index++) {

			int start_Y_initial = start_Y;

			for(int i=0; i<NUM_STREAMS; i++) {
				hipStreamSynchronize(hipStreamToUse[i]);
			}

			/* Combines an SNP X with a block of SNPs Y. */
			dim3 blocksPerGrid_prework_k3( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1)), 1, 1);
			dim3 workgroupSize_prework_k3( 1, 128, 1 );
			epistasis_prework_k3<<<blocksPerGrid_prework_k3, workgroupSize_prework_k3, 0, hipStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), cases_A_block16bit_ptrGPU, controls_A_block16bit_ptrGPU, numSNPs, numCases, numControls, X_index, start_Y_initial);	
			
			/* Calculates pairwise population counts between a single SNP X and all SNPs with index larger or equal to 'start_Y'.
			   Takes into account only the blocks of SNPs that are going to accessed in the nested loop */
			dim3 blocksPerGrid_pairwise_singleX_Z ( (size_t)ceil(((float)(numSNPs - start_Y)) / ((float)1)), 1, 1);	
			dim3 workgroupSize_pairwise_singleX_Z ( 1, 128, 1 );
			hipLaunchKernelGGL(epistasis_pairwiseSNPs_singleX_Z, blocksPerGrid_pairwise_singleX_Z, workgroupSize_pairwise_singleX_Z, 0, hipStream_singleX_Z, (uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls, X_index, start_Y);  

			for(int start_Z = start_Y; start_Z < numSNPsWithPadding; start_Z+=BLOCK_SIZE) {
				(*roundsCounter)++;


				/* Converts data from a block of SNPs from 1-bit to 16-bit */

                                __half *cases_B_block16bit_ptrGPU_forStream = (__half *) (cases_B_block16bit_ptrGPU + (objectiveFunctionIndex % NUM_STREAMS) * numCasesWithPadding * (SNP_CALC) * BLOCK_SIZE);
                                __half *controls_B_block16bit_ptrGPU_forStream = (__half *) (controls_B_block16bit_ptrGPU + (objectiveFunctionIndex % NUM_STREAMS) * numControlsWithPadding * (SNP_CALC) * BLOCK_SIZE);

				dim3 blocksPerGrid_convertBinaryBlockTo16bit( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1)), 1, 1);
				dim3 workgroupSize_convertBinaryBlockTo16bit( 1, 128, 1 );

				epistasis_convertBinaryBlockTo16bit<<<blocksPerGrid_convertBinaryBlockTo16bit, workgroupSize_convertBinaryBlockTo16bit, 0, hipStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), cases_B_block16bit_ptrGPU_forStream, controls_B_block16bit_ptrGPU_forStream, numSNPs, numCases, numControls, start_Z);     


                                /* Processes Cases */

                                hipblasStatus_t hipblasReturn = hipblasGemmEx(hipblasHandlers[(objectiveFunctionIndex % NUM_STREAMS)], transa, transb, BLOCK_SIZE * SNP_CALC * SNP_CALC, BLOCK_SIZE * SNP_CALC, numCasesWithPadding, &alpha, 
                                                cases_A_block16bit_ptrGPU, // d_A 
                                                HIPBLAS_R_16F, A_leadingDim_cases, 
                                                cases_B_block16bit_ptrGPU_forStream, // d_B 
                                                HIPBLAS_R_16F, B_leadingDim_cases, &beta, 
                                                C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),       // d_C
                                                HIPBLAS_R_32F, C_leadingDim, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);

                                if(hipblasReturn != HIPBLAS_STATUS_SUCCESS) {
                                        printf("problem executing hipBLAS\n");
                                }


                                /* Processes Controls */

                                hipblasReturn = hipblasGemmEx(hipblasHandlers[(objectiveFunctionIndex % NUM_STREAMS)], transa, transb, BLOCK_SIZE * SNP_CALC * SNP_CALC, BLOCK_SIZE * SNP_CALC, numControlsWithPadding, &alpha,
                                                controls_A_block16bit_ptrGPU, // d_A 
                                                HIPBLAS_R_16F, A_leadingDim_controls,
                                                controls_B_block16bit_ptrGPU_forStream, // d_B 
                                                HIPBLAS_R_16F, B_leadingDim_controls, &beta,
                                                C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),       // d_C
                                                HIPBLAS_R_32F, C_leadingDim, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);


				if(hipblasReturn != HIPBLAS_STATUS_SUCCESS) {
					printf("problem executing hipBLAS\n");
				}

				if(X_index == 0) {
					hipStreamSynchronize(hipStreamPairwiseSNPs);
				}

				if(start_Z == start_Y)  {
					hipStreamSynchronize(hipStream_singleX_Z);
				}


	                       /* Derives contingency tables from the output of the matrix operations and calculates objective scoring function.
        	                  The reduction of scores and identification of the best candidate solution is also implemented by the same kernel. */

                               dim3 blocksPerGrid_objFun( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)64) ), (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1) / ((float)BLOCK_OBJFUN)), 1);
				dim3 workgroupSize_objFun( 64, 1, 1 );

				
				if((start_Z + BLOCK_SIZE) > numSNPs) {
                                	objectiveFunctionKernel<true><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, hipStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_packedIndices, start_Y, start_Z, X_index, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls);
				}
				else {
					objectiveFunctionKernel<false><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, hipStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_packedIndices, start_Y, start_Z, X_index, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls);
				}

				objectiveFunctionIndex++;
			}
		}
	}

	/* In case evaluation of blocks is still ongoing */
	for (int i = 0; i < NUM_STREAMS; i++) {
		hipStreamSynchronize(hipStreamToUse[i]);
	}

	/* Copies best solution found from GPU memory to Host */
	hipMemcpy(outputFromGpu, d_output, sizeof(float), hipMemcpyDeviceToHost);
	hipMemcpy(output_indexFromGpu_packedIndices, d_output_packedIndices, sizeof(unsigned long long int), hipMemcpyDeviceToHost);


	hipDeviceSynchronize();	
	clock_gettime(CLOCK_MONOTONIC, &t_end); // final timestamp

        (*searchTime) = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));


	hipFree(cases_A_ptrGPU);
	hipFree(C_ptrGPU_cases);
	hipFree(controls_A_ptrGPU);
	hipFree(C_ptrGPU_controls);
	hipFree(d_output_individualSNP_popcountsForControls);
	hipFree(d_output_individualSNP_popcountsForCases);
	hipFree(d_tablePrecalc);
	hipFree(d_output);
	hipFree(d_output_packedIndices);

	free(h_tablePrecalc);
	
	return hipSuccess;
}
