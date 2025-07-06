/**
 *
 * mpi-tensor-episdet: Multi-GPU and multi-node high-throughput high-order exhaustive epistasis detection using matrix multiply operations on AMD GPU matrix cores
 *
 * High-Performance Computing Architectures and Systems (HPCAS) Group, INESC-ID

 * Contact: Ricardo Nobre <ricardo.nobre@inesc-id.pt>
 *
 */

// Standard Library
#include <iostream>
#include <iomanip>      
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
#include <libgen.h>
#include <mpi.h>

#include "helper_amd.hpp"
#include "reduction_amd.hpp"

#if defined(TRIPLETS)
#include "search-triplets_amd.hpp"
#else
#include "search-pairs.hpp"
#endif

#define MAX_CHAR_ARRAY 1000


int main(int argc, const char *arg[]) {


	if(argc < 2) {
		std::cerr << "Usage: tensor-episdet dataset.txt" << std::endl;
		return 1;
	}


	/* Reads information about input dataset. */

        FILE* fStream = fopen(arg[1], "r");     		// File with information and pointers to dataset.
	if(fStream == NULL) {
		std::cerr << "File '" << arg[1] << "' does not exist!" << std::endl;
		return 1;
	}

	char* ts = strdup(arg[1]);
	char* pathToDataset = dirname(ts);	

        char line[MAX_CHAR_ARRAY];	
        char* ret = fgets(line, MAX_CHAR_ARRAY, fStream); 	// First line represents the number of SNPs.
        uint numSNPs = atoi(line);

	char controlsFileName[MAX_CHAR_ARRAY];
        ret = fgets(controlsFileName, MAX_CHAR_ARRAY, fStream);	// Second line represents the filename with controls data.
	controlsFileName[strcspn(controlsFileName, "\n")] = 0;	// Removes trailing newline character.

        ret = fgets(line, MAX_CHAR_ARRAY, fStream); 		// Third line represents the number of controls.
        uint numControls = atoi(line);

	char casesFileName[MAX_CHAR_ARRAY];
        ret = fgets(casesFileName, MAX_CHAR_ARRAY, fStream);	// Forth line represents the filename with cases data.
	casesFileName[strcspn(casesFileName, "\n")] = 0;	// Removes trailing newline character.

        ret = fgets(line, MAX_CHAR_ARRAY, fStream); 		// Fifth line represents the number of cases.
        uint numCases = atoi(line);


	/* Calculates number of distinct blocks and padds number of SNPs to process to the block size. */
	uint numBlocks = ceil((float)numSNPs / (float)BLOCK_SIZE);
	uint numSNPsWithPadding = numBlocks * BLOCK_SIZE;

	/* Padds the number of controls and of cases. */
	uint numCasesWithPadding = ceil((float)numCases / PADDING_SAMPLES) * PADDING_SAMPLES;	
	uint numControlsWithPadding = ceil((float)numControls / PADDING_SAMPLES) * PADDING_SAMPLES;


	/* Prints information about dataset and number of distinct blocks of SNPs to process. */
        std::cout << "Num. SNPs: " << numSNPs << std::endl;
        std::cout << "Num. Blocks of SNPs: " << numBlocks << std::endl;
        std::cout << "Num. Cases: " << numCases << std::endl;
        std::cout << "Num. Controls: " << numControls << std::endl;


	/* Allocates pinned memory for holding controls and cases dataset matrices.
	   Each 32-bit 'unsigned int' holds 32 binary values representing genotype information.
 	   Only two allele types are represented (SNP_CALC macro equals 2), ...
	   ... being information about the third allele type infered.
	 */
	
	int numSamplesCases_32packed = ceil(((float) numCasesWithPadding) / 32.0f);
	int numSamplesControls_32packed = ceil(((float) numControlsWithPadding) / 32.0f);

        int datasetCases_32packed_size = numSamplesCases_32packed * numSNPsWithPadding * SNP_CALC;
        unsigned int* datasetCases_32packed_matrixA = NULL;
        hipError_t result = hipHostAlloc((void**)&datasetCases_32packed_matrixA, datasetCases_32packed_size * sizeof(unsigned int), hipHostMallocDefault );     
        if(datasetCases_32packed_matrixA == NULL) {
                std::cerr << "Problem allocating Host memory for cases" << std::endl;
        }

        int datasetControls_32packed_size = numSamplesControls_32packed * numSNPsWithPadding * SNP_CALC;    
        unsigned int* datasetControls_32packed_matrixA = NULL;
        result = hipHostAlloc((void**)&datasetControls_32packed_matrixA, datasetControls_32packed_size * sizeof(unsigned int), hipHostMallocDefault );
        if(datasetControls_32packed_matrixA == NULL) {
                std::cerr << "Problem allocating Host memory for controls" << std::endl;
        }


	/* Reads dataset (controls and cases data) from storage device.
	   Input dataset must be padded with zeros in the dimension of samples (cases / controls), ...
	   ... making the number of bits per {SNP, allele} tuple a multiple of PADDING_SAMPLES. */

	size_t numElem;
	std::string absolutePathToCasesFile = std::string(pathToDataset) + "/" + casesFileName;
	FILE *ifp_cases = fopen(absolutePathToCasesFile.c_str(), "rb");
	numElem = fread(datasetCases_32packed_matrixA, sizeof(unsigned int), numSamplesCases_32packed * numSNPs * SNP_CALC, ifp_cases);
	if(numElem != datasetCases_32packed_size) {
		std::cerr << "Problem loading cases from storage device" << std::endl;
	}
	fclose(ifp_cases);

	std::string absolutePathToControlsFile = std::string(pathToDataset) + "/" + controlsFileName;
	FILE *ifp_controls = fopen(absolutePathToControlsFile.c_str(), "rb");
	numElem = fread(datasetControls_32packed_matrixA, sizeof(unsigned int), numSamplesControls_32packed * numSNPs * SNP_CALC, ifp_controls);
        if(numElem != datasetControls_32packed_size) {
                std::cerr << "Problem loading controls from storage device" << std::endl;
        }
	fclose(ifp_controls);

	std::cout << "-------------------------------" << std::endl;


        /* Initializes the MPI environment */
        MPI_Init(NULL, NULL);

        /* Gets the number of processes */
        int mpi_world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

        /* Gets the rank of the calling process */
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


	int start_Y = 0;

	int numActiveSlaves = mpi_world_size - 1;	

        float outputFromGpu = FLT_MAX;
        unsigned long long int output_indexFromGpu_packedIndices;

	if (mpi_world_rank == 0) {

		MPI_Status mpiStatus;

		while ( numActiveSlaves > 0 ) {

			MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0 /* only one type of tag */, MPI_COMM_WORLD, &mpiStatus);
			int slaveRank = mpiStatus.MPI_SOURCE;


			if (start_Y < numSNPsWithPadding) {

				MPI_Send(&start_Y, 1, MPI_INT , slaveRank, 0 /* only one type of tag */, MPI_COMM_WORLD);
				start_Y += BLOCK_SIZE;

			} else {

				int stopSlave = -1;
				MPI_Send (&stopSlave, 1, MPI_INT, slaveRank , 0 /* only one type of tag */ , MPI_COMM_WORLD);

				numActiveSlaves--;
			}
		}


	} else {


		/* Launches epistasis detection search. */

		int roundsCounter;	
		double searchTime;	

		result = EpistasisDetectionSearch(
				datasetCases_32packed_matrixA,		// Cases matrix.
				datasetControls_32packed_matrixA,	// Controls matrix.
				numSNPs,                                // Number of SNPs.
				numCases,                               // Number of cases.
				numControls,                            // Number of controls.
				numSNPsWithPadding,                     // Number of SNPs padded to block size.
				numCasesWithPadding,     		// Number of cases padded to PADDING_SIZE.
				numControlsWithPadding,     		// Number of controls padded to PADDING_SIZE.
				&roundsCounter,				// Counter for number of rounds processed.
				&searchTime,				// Counter for execution time (seconds).
				&outputFromGpu,				// Score of best score found.
				&output_indexFromGpu_packedIndices	// Indexes of SNPs of set that results in best score.
				);

		if(result != hipSuccess) {
			std::cerr << "Epistasis detection search failed." << std::endl;
		}

	}

	/* Reduces scores and gets indices of best score */

	float * outputFromGpuPerMpiProcess; 
        unsigned long long int * output_indexFromGpu_packedIndicesPerMpiProcess;

    	if ( mpi_world_rank == 0) { 
       		outputFromGpuPerMpiProcess = (float *)malloc(mpi_world_size*1*sizeof(float)); 
                output_indexFromGpu_packedIndicesPerMpiProcess = (unsigned long long int *)malloc(mpi_world_size*1*sizeof(unsigned long long int));

       	} 

	MPI_Gather( &outputFromGpu, 1, MPI_FLOAT, outputFromGpuPerMpiProcess, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);	
	MPI_Gather( &output_indexFromGpu_packedIndices, 1, MPI_UNSIGNED_LONG_LONG, output_indexFromGpu_packedIndicesPerMpiProcess, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);    


	float outputFromGpuAllMpiProcesses = FLT_MAX;
	unsigned long long output_indexFromGpu_packedIndicesAllMpiProcesses;

	if(mpi_world_rank == 0) {
		for(int i = 0; i < mpi_world_size; i++) {
			if(outputFromGpuPerMpiProcess[i] < outputFromGpuAllMpiProcesses) {
				outputFromGpuAllMpiProcesses = outputFromGpuPerMpiProcess[i];
				output_indexFromGpu_packedIndicesAllMpiProcesses = output_indexFromGpu_packedIndicesPerMpiProcess[i];
			}
		}

	}


	MPI_Barrier(MPI_COMM_WORLD); 
        clock_gettime(CLOCK_MONOTONIC, &t_end);

	MPI_Finalize();


	if(mpi_world_rank == 0) {

                /* Prints indices of set of SNPs most associated with phenotype and its score */

		double timing_duration_mpi = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

		#if defined(TRIPLETS)
		std::cout << "-------------------------------" << std::endl << "{SNP_X_i, SNP_Y_i, SNP_Z_i}: SCORE\t->\t{" << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 0) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 21) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 42) & 0x1FFFFF) << "}: " << std::fixed << std::setprecision(6) << outputFromGpuAllMpiProcesses << std::endl;
		#else
		std::cout << "-------------------------------" << std::endl << "{SNP_X_i, SNP_Y_i}: SCORE\t->\t{" << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 0) & 0xFFFFFFFF) << ", " << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 32) & 0xFFFFFFFF) << "}: " << std::fixed << std::setprecision(6) << outputFromGpuAllMpiProcesses << std::endl;
		#endif

                /* Prints time to execute the application, the achieved performance, and the ratio of unique sets */

		unsigned long long numCombinations = n_choose_k(numSNPs, INTER_OR);

		std::cout << "Wall-clock time:\t" << std::fixed << std::setprecision(3) << timing_duration_mpi << " seconds" << std::endl;

		std::cout << "Num. unique sets per sec. (scaled to sample size): " << std::fixed << std::setprecision(3) << (((double) numCombinations * (double) (numCases + numControls) / (double)(timing_duration_mpi)) / 1e12) << " x 10^12" << std::endl;

		std::cout << "Unique sets of SNPs evaluated (k=" << INTER_OR << "): " << numCombinations << std::endl;

	}

	return result == hipSuccess ? 0 : 1;	

}






