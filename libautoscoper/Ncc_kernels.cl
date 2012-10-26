__kernel
void sum_kernel(
		__global float* f,
		__global float* sums,
		unsigned int n)
{
	extern __shared__ float sdata[];

	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	sdata[threadIdx.x] = (i < n) ? f[i] : 0.0f;

	__syncthreads();
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		sums[blockIdx.x] = sdata[0];
	}
}

__kernel
void ncc_kernel(
		__global float* f,
		float meanF,
		__global float* g,
		float meanG,
		__global float* nums,
		__global float* den1s,
		_global float* den2s,
		unsigned int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	if (i < n) {
		float fMinusMean = f[i]-meanF;
		float gMinusMean = g[i]-meanG;

		nums[i] = fMinusMean*gMinusMean;
		den1s[i] = fMinusMean*fMinusMean;
		den2s[i] = gMinusMean*gMinusMean;
	}
	else {
		nums[i] = 0.0f;
		den1s[i] = 0.0f;
		den2s[i] = 0.0f;
	}
}

// vim: ts=4 syntax=cpp noexpandtab
