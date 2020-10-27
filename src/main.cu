#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <typeinfo>

// Include the header for device random number generation.
#include <curand.h>
#include <curand_kernel.h>

#include "CUDAErrorChecking.cuh"
#include "NVTXUtil.cuh"
#include "CUDAEventTimer.cuh"


/** 
 * Curand generator types
 * ## Psuedo RNG
 * CURAND_RNG_PSEUDO_XORWOW        curandStateXORWOW_t
 * CURAND_RNG_PSEUDO_MRG32K3A      curandStateMRG32k3a_t
 * CURAND_RNG_PSEUDO_MTGP32        curandStateMtgp32_t
 * CURAND_RNG_PSEUDO_PHILOX4_32_10 curandStatePhilox4_32_10_t
 * 
 * ## Host API only  
 * CURAND_RNG_PSEUDO_MT19937
 *
 * ## Quasi
 * CURAND_RNG_QUASI_SOBOL32           curandStateSobol32_t
 * CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 curandStateScrambledSobol32_t
 * CURAND_RNG_QUASI_SOBOL64           curandStateSobol64_t
 * CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 curandStateScrambledSobol64_t
 */

template<typename T>
__global__ void curand_initialise(const unsigned int THREADS, const unsigned long long int seed, T* curandStates) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < THREADS) {
        curand_init(seed, idx, 0, &curandStates[idx]);
    }
}

// @todo - kernel which does memory accesses but doesn't use the kernel for overhead measurement.



// @note - risk of it being optimised out as not using the result. 
// @todo double variants
// @todo normal / lognormal
// @todo float2/float4/double2 variants.
template<typename T>
__global__ void curand_uniformf_sample(const unsigned int THREADS, const unsigned int SAMPLES, T* curandStates) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < THREADS) {
        // Load state into register.
        T state = curandStates[idx];
        float value = 0.f;
        for(unsigned int sampleIdx = 0; sampleIdx < SAMPLES; sampleIdx++){
            value = curand_uniform(&state);
            // @todo make sure this isnt' being optimised out.
        }
        // Update the state in global memory.
        curandStates[idx] = state;
    }
}


template<typename T>
bool allocate(const unsigned int THREADS, T ** d_curandStates, size_t * allocatedBytes){
    NVTX_RANGE("allocate");
    
    bool success = true;
    size_t bytes = THREADS * sizeof(T);

    CUDA_CALL(cudaMalloc((void**)d_curandStates, bytes));     
    *allocatedBytes = bytes;

    return success;
}

template<typename T>
void deallocate(T ** d_curandStates){
    CUDA_CALL(cudaFree(*d_curandStates));
    *d_curandStates = nullptr;
}

template <typename T>
void curand_bench(const unsigned int THREADS, const unsigned int SAMPLES, const unsigned int REPS, const bool AGGREGATE_OUTPUT){
    NVTX_RANGE(typeid(T).name());

    size_t totalCurandStateBytes = 0;
    double totalAllocMillis = 0.;
    double totalInitMillis = 0.;
    double totalUniformfSampleMillis = 0.;
    double totalDeallocMillis = 0.;


    // Repeat a few times to end up with better timings.
    for(unsigned int rep = 0; rep < REPS; rep++){
        // Prep timers
        CUDAEventTimer allocTimer = CUDAEventTimer();
        CUDAEventTimer initTimer = CUDAEventTimer();
        CUDAEventTimer uniformfSampleTimer = CUDAEventTimer();
        CUDAEventTimer deallocTimer = CUDAEventTimer();

        // prep occupancy
        int gridSize = 0;
        int minGridSize = 0;
        int blockSize = 0;

        // prep storage
        T * d_curandStates = nullptr; 
        size_t curandStateBytes = 0;

        // Prep the seed
        const unsigned long long int seed = rep;


        // Allocate curand (and time) 
        allocTimer.start();
        allocate(THREADS, &d_curandStates, &curandStateBytes);
        allocTimer.stop();
        allocTimer.sync();

        // Initialise curand (and time)
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, curand_initialise<T>, 0, 0));
        gridSize = (THREADS + blockSize - 1) / blockSize; 

        initTimer.start();
        curand_initialise<T><<< gridSize, blockSize >>>(THREADS, seed, d_curandStates); 
        initTimer.stop();
        initTimer.sync();


        // Sample from curand (and time)
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, curand_uniformf_sample<T>, 0, 0));
        gridSize = (THREADS + blockSize - 1) / blockSize; 

        uniformfSampleTimer.start();
        curand_uniformf_sample<T><<< gridSize, blockSize >>>(THREADS, SAMPLES, d_curandStates); 
        uniformfSampleTimer.stop();
        uniformfSampleTimer.sync();


        // Free (dont' time)
        deallocTimer.start();
        deallocate(&d_curandStates);
        deallocTimer.stop();
        deallocTimer.sync();

        // Output data for this run. as csv.
        float allocMillis = allocTimer.getElapsedMilliseconds();
        float initMillis = initTimer.getElapsedMilliseconds();
        float uniformfSampleMillis = uniformfSampleTimer.getElapsedMilliseconds();
        float deallocMillis = deallocTimer.getElapsedMilliseconds();

        if (!AGGREGATE_OUTPUT) {
            // event timers have 0.5us resolution, so %.4f ms
            printf(
                "%s,%u,%u,%llu,%zu,%.4f,%.4f,%.4f,%.4f\n", 
                typeid(T).name(),
                THREADS, 
                SAMPLES,
                seed,
                curandStateBytes,
                allocMillis,
                initMillis,
                uniformfSampleMillis,
                deallocMillis
            );
        } else {
            totalCurandStateBytes += curandStateBytes;
            totalAllocMillis += allocMillis;
            totalInitMillis += initMillis;
            totalUniformfSampleMillis += uniformfSampleMillis;
            totalDeallocMillis += deallocMillis;
        }
    }
    if (AGGREGATE_OUTPUT) {
        // event timers have 0.5us resolution, so %.4f ms
        printf(
            "%s,%u,%u,%u,%f,%.4f,%.4f,%.4f,%.4f\n", 
            typeid(T).name(),
            THREADS, 
            SAMPLES,
            REPS,
            totalCurandStateBytes / (float)REPS,
            totalAllocMillis / (float)REPS,
            totalInitMillis / (float)REPS,
            totalUniformfSampleMillis / (float)REPS,
            totalDeallocMillis / (float)REPS
        );
    }
}

bool benchmark(const unsigned int THREADS, const unsigned int SAMPLES, const unsigned int REPS, const bool AGGREGATE_OUTPUT){
    // Push a range marker.
    NVTX_RANGE("benchmark");

    // print the header
    if(AGGREGATE_OUTPUT){
        printf("engine,threads,samples_per_thread,repetitions,mean_bytes,mean_alloc_ms,mean_init_ms,mean_uniformf_ms,mean_dealloc_ms\n");
    } else {
        printf("engine,threads,samples_per_thread,seed,bytes,alloc_ms,init_ms,uniformf_ms,dealloc_ms\n");
    }

    // Xorwow
    curand_bench<curandStateXORWOW_t>(THREADS, SAMPLES, REPS, AGGREGATE_OUTPUT);
    // MRG
    curand_bench<curandStateMRG32k3a_t>(THREADS, SAMPLES, REPS, AGGREGATE_OUTPUT);
    // MTG - requires special initialisation + a syncthreads per block, so not worth considering for our use-case.
    // curand_bench<curandStateMtgp32_t>(THREADS, SAMPLES, REPS, AGGREGATE_OUTPUT);
    // Philox
    curand_bench<curandStatePhilox4_32_10_t>(THREADS, SAMPLES, REPS, AGGREGATE_OUTPUT);

    return true;
}

void cudaInit(){
    NVTX_RANGE("cudaInit");
    // Free the nullptr to initialise the cuda context.
    CUDA_CALL(cudaFree(0));
}

int main(int argc, char * argv[]){
    // Early initialise the cuda context to improve profiling clarity.
    cudaInit();

    // @todo - move to args / make sure enough to fully saturate the device.
    // Probably better to ask for N samples in total, and calc threads based on that (or use a grid strided loop + full device launch.)
    const unsigned int THREADS = 262144;
    const unsigned int SAMPLES = 65536;
    // const unsigned int SAMPLES = 1048576;
    const unsigned int REPS = 5;
    const bool AGGREGATE_OUTPUT = true;

    // Run some stuff.
    bool success = benchmark(THREADS, SAMPLES, REPS, AGGREGATE_OUTPUT);

    // Reset the device.
    cudaDeviceReset();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
