#ifndef INCLUDE_CUDAERRORCHECKING_CUH_
#define INCLUDE_CUDAERRORCHECKING_CUH_

static void HandleCUDAError(const char *file, int line, cudaError_t status = cudaGetLastError()) {
	if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess)
	{
		if (status == cudaErrorUnknown)
		{
			printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
			exit(1);
		}
		printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
		exit(1);
	}
}

#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))


#endif  // INCLUDE_CUDAERRORCHECKING_CUH_
