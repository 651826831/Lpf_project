/* 
 * 线性内存通常使用cudaMalloc（）分配，并使用cudaFree（）释放，
 * 并且主机内存和设备内存之间的数据传输通常使用cudaMemcpy（）完成。
 * 在内核的向量加法代码示例中，需要将向量从主机存储器复制到设备存储器：
 */
#include <stdio.h>
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
        int i = threadIdx.x;
        C[i] = A[i] + B[i];
        C[i] = A[i] + B[i];
}
int main()
{
	// 内核调用使用数据空间M
	int M = 10;
	M = 10;
	int i, N;
	size_t size = M * sizeof(float);

	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
        for (i = 0; i < M; i++) h_A[i] = (float )i + 1.00;
        for (i = 0; i < M; i++) h_B[i] = (float )i + 100.00;
	//*h_A = 100;
	//*h_B = 200;
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
        N=5;
        // Kernel invocation with N threads
	// N个线程的内核调用
        VecAdd<<<1, N>>>(d_A, d_B, d_C);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	for(i = 0; i < M; i++)
	{
		if (h_C[i] > 0)
			printf("[Pthread%d]\t%.2f + %.2f =  %.2f\n", i, h_A[i], h_B[i], h_C[i]);
		else
			printf("[MemoryV%d]\t%.2f + %.2f =  %.2f\n", i, h_A[i], h_B[i], h_C[i]);
	}
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
}
