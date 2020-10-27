#ifndef INCLUDE_CUDAEVENTTIMER_CUH_
#define INCLUDE_CUDAEVENTTIMER_CUH_

#include "CUDAErrorChecking.cuh"

/**
 * Class to simplify the use of CUDAEvents for timing.
 * Timing between CUDAEvent_t is only accurate in the default stream, hence streams cannot be passed.
 */
class CUDAEventTimer {
 public:
    /** 
     * Default constructor, creates the cudaEvents and initialises values.
     */
    CUDAEventTimer() :
    startEvent(NULL),
    stopEvent(NULL),
    ms(0.),
    synced(false) {
        CUDA_CALL(cudaEventCreate(&this->startEvent));
        CUDA_CALL(cudaEventCreate(&this->stopEvent));
    }
    /** 
     * Destroys the cudaEvents created by this instance
     */
    ~CUDAEventTimer() {
        CUDA_CALL(cudaEventDestroy(this->startEvent));
        CUDA_CALL(cudaEventDestroy(this->stopEvent));
        this->startEvent = NULL;
        this->startEvent = NULL;
    }
    /**
     * Record the start event, resetting the syncronisation flag.
     */
    void start() {
        CUDA_CALL(cudaEventRecord(this->startEvent));
        synced = false;
    }
    /**
     * Record the stop event, resetting the syncronisation flag.
     */
    void stop() {
        CUDA_CALL(cudaEventRecord(this->stopEvent));
        synced = false;
    }
    /**
     * Syncrhonize the cudaEvent(s), calcualting the elapsed time in ms between the two events.
     * this is only accurate if used for the default stream (hence streams are not used)
     * Sets the flag indicating syncronisation has occured, and therefore the elapsed time can be queried.
     * @return elapsed time in milliseconds
     */
    float sync() {
        CUDA_CALL(cudaEventSynchronize(this->stopEvent));
        CUDA_CALL(cudaEventElapsedTime(&this->ms, this->startEvent, this->stopEvent));
        synced = true;
        return ms;
    }
    /**
     * Get the elapsed time between the start event being issued and the stop event occuring.
     * @return elapsed time in milliseconds
     */
    float getElapsedMilliseconds() {
        if (!synced) {
            printf("Error: Unsynced CUDAEventTimer.\n");
            exit(EXIT_FAILURE);
        }
        return ms;
    }

 private:
    /**
     * CUDA Event for the start event
     */
    cudaEvent_t startEvent;
    /**
     * CUDA Event for the stop event
     */
    cudaEvent_t stopEvent;
    /**
     * Elapsed times between start and stop in milliseconds
     */
    float ms;
    /**
     * Flag to return whether events have been synced or not.
     */
    bool synced;
};

#endif  // INCLUDE_CUDAEVENTTIMER_CUH_
