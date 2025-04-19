// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  return optimal_block_num;
//   int max_block_num = 65000;
//   return min(optimal_block_num, max_block_num);
}


// template <typename scalar_t>
__device__ void get_gate_idx_sf(int h1, int w1, int h2, int w2, int * out)
{
    if(w1>w2)
    {
        out[0]=h1;
        out[1]=w1;
    }
    else
    {
        out[0]=h2;
        out[1]=w2;
    }
}


template <typename scalar_t>
__device__ scalar_t get_data_sf(const scalar_t * data, 
								int num, int channels, 
								int height, int min_height, int max_height, 
								int width, int min_width, int max_width, 
								int n, int c, int h, int w)
{
	if(h<min_height || h>=max_height)
		return 0;
	if(w<min_width || w>=max_width)
		return 0;

	return data[n*channels*height*width + c * height*width + h * width + w];
}


template <typename scalar_t>
__device__ void set_data_sf(scalar_t * data, 
							int num, int channels, 
							int height, int min_height, int max_height, 
							int width, int min_width, int max_width, 
							int n, int c, int h, int w, 
							scalar_t v)
{
	if(h<min_height || h>=max_height)
		return ;
	if(w<min_width || w>=max_width)
		return ;

	data[n*channels*height*width + c * height*width + h * width + w]=v;
}


template <typename scalar_t>
__device__ scalar_t get_gate_sf(const scalar_t * data, 
								int num, int channels, 
								int height, int min_height, int max_height, 
								int width, int min_width, int max_width, 
								int n, int c, int h1, int w1, int h2, int w2)
{
	if(h1<min_height || h1>=max_height)
		return 0;
	if(w1<min_width || w1>=max_width)
		return 0;
	if(h2<min_height || h2>=max_height)
		return 0;
	if(w2<min_width || w2>=max_width)
		return 0;
	int idx[2];

	get_gate_idx_sf(h1,w1,h2,w2, idx);

	int h = idx[0];
	int w = idx[1];

	return data[n*channels*height*width + c * height*width + h * width + w];
}


template <typename scalar_t>
__device__ void set_gate_sf(scalar_t * data, 
							int num, int channels, 
							int height, int min_height, int max_height, 
							int width, int min_width, int max_width, 
							int n, int c, int h1, int w1, int h2, int w2, 
							scalar_t v)
{
	if(h1<min_height || h1>=max_height)
		return ;
	if(w1<min_width || w1>=max_width)
		return ;
	if(h2<min_height || h2>=max_height)
		return ;
	if(w2<min_width || w2>=max_width)
		return ;
	int idx[2];
  	get_gate_idx_sf(h1,w1,h2,w2,idx);

	int h = idx[0];
	int w = idx[1];

	data[n*channels*height*width + c * height*width + h * width + w]=v;
}

// we do not use set_gate_add_sf(...) in the caffe implimentation
// avoid using atomicAdd


template <typename scalar_t>
__global__ void forward_one_chunk_four_directions(int count, int t, int kchunk, int nitems, int num, int channels, int height, int width, const scalar_t* X, const scalar_t* B, const scalar_t* G1,  const scalar_t* G2, const scalar_t* G3, scalar_t* H) {
  CUDA_1D_KERNEL_LOOP(index, count) {
  	int khc_count = kchunk * height * channels;
	int kh_count = kchunk * height;

  	int n,c,h,k,w;
  	int temp=index;
	int MIN_w, MAX_w, MIN_h, MAX_h;

  	n = temp / khc_count;
  	temp = temp % khc_count;
  	c = temp / kh_count;
  	temp = temp % kh_count;
  	h = temp / kchunk;
  	temp = temp % kchunk;
  	k = temp;

	w = k * nitems + t; 

	MIN_w = k * nitems;
	MAX_w = min((k + 1) * nitems, width);
	MIN_h = 0;
	MAX_h = height;

	scalar_t x_data = get_data_sf(X,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w);
	scalar_t b_data = get_data_sf(B,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w);

	scalar_t g_data_1 = get_gate_sf(G1,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h-1,w-1);
	scalar_t h_minus1_data_1 = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h-1,w-1);
	scalar_t h1_minus1 = g_data_1 * h_minus1_data_1;

	scalar_t g_data_2 = get_gate_sf(G2,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h,w-1);
	scalar_t h_minus1_data_2 = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w-1);
	scalar_t h2_minus1 = g_data_2 * h_minus1_data_2;

	scalar_t g_data_3 = get_gate_sf(G3,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h+1,w-1);
	scalar_t h_minus1_data_3 = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h+1,w-1);
	scalar_t h3_minus1 = g_data_3 * h_minus1_data_3;

	scalar_t h_hype = h1_minus1 + h2_minus1 + h3_minus1;
	scalar_t x_hype = b_data * x_data;

	scalar_t h_data = x_hype + h_hype;

	set_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h_data);
  }
}


template <typename scalar_t>
__global__ void backward_one_chunk_four_directions(int count, int t, int kchunk, int nitems, int num,int channels, int height, int width, const scalar_t* X, const scalar_t* B, const scalar_t* G1, const scalar_t* G2, const scalar_t* G3, const scalar_t* H, scalar_t* X_diff, scalar_t* Bdiff, scalar_t* G1_diff, scalar_t* G2_diff, scalar_t* G3_diff, scalar_t* Hdiff) {
  CUDA_1D_KERNEL_LOOP(index, count) {
  	int khc_count = kchunk * height * channels;
	int kh_count = kchunk * height;

  	int n,c,h,k,w;
  	int temp=index;
	int MIN_w, MAX_w, MIN_h, MAX_h;

  	n = temp / khc_count;
  	temp = temp % khc_count;
  	c = temp / kh_count;
  	temp = temp % kh_count;
  	h = temp / kchunk;
  	temp = temp % kchunk;
  	k = temp;

	w = (width - 1 - k * nitems) - t; 

	MIN_w = max(0, width - (k + 1) * nitems);
	MAX_w = width - k * nitems;
	MIN_h = 0;
	MAX_h = height;

	scalar_t x_data = get_data_sf(X,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w);
	scalar_t b_data = get_data_sf(B,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w);

	//h(t)_diff = top(t)_diff
	scalar_t h_diff = get_data_sf(Hdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w);

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	scalar_t add1_h3_diff = get_data_sf(Hdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h-1,w+1);
	scalar_t add1_g3_data = get_gate_sf(G3,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h-1,w+1);

	scalar_t add1_h2_diff = get_data_sf(Hdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w+1);
	scalar_t add1_g2_data = get_gate_sf(G2,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h,w+1);

	scalar_t add1_h1_diff = get_data_sf(Hdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h+1,w+1);
	scalar_t add1_g1_data = get_gate_sf(G1,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h+1,w+1);

	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;

	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data_sf(Hdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h_diff);

	//x(t)_diff = B * h(t)_diff
	scalar_t x_diff = b_data * h_diff; // modified
	set_data_sf(X_diff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,x_diff);

	// lambda_diff = h(t)_diff * x(t)
	scalar_t b_diff = x_data * h_diff; // added
	set_data_sf(Bdiff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,b_diff);

	// g_diff = h_diff * h_data(t-1)
	scalar_t h1_minus1_data = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h-1,w-1);
	scalar_t g1_diff = h_diff * h1_minus1_data;
	set_gate_sf(G1_diff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h-1,w-1,g1_diff);

	scalar_t h2_minus1_data = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w-1);
	scalar_t g2_diff = h_diff * h2_minus1_data;
	set_gate_sf(G2_diff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h,w-1,g2_diff);

	scalar_t h3_minus1_data = get_data_sf(H,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h+1,w-1);
	scalar_t g3_diff = h_diff * h3_minus1_data;
	set_gate_sf(G3_diff,num,channels,height,MIN_h,MAX_h,width,MIN_w,MAX_w,n,c,h,w,h+1,w-1,g3_diff);
  }
}


// fwd kernel 
int Forward_four_directions(int num_, int channels_, int height_, int width_, int kNItems_,
						const at::Tensor X, 
						const at::Tensor B,
						const at::Tensor G1, const at::Tensor G2, const at::Tensor G3, 
						at::Tensor H)
{
  int kNItems = min(width_, kNItems_);
  int kchunk = (width_ + kNItems - 1) / kNItems;
  int count = kchunk * height_ * channels_ * num_;

  if (X.scalar_type() == at::ScalarType::Float)  {
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		X.scalar_type(), "Forward_four_directions_gpu", ([&] {
			const scalar_t *X_data = X.data<scalar_t>();
			const scalar_t *B_data = B.data<scalar_t>();
			const scalar_t *G1_data = G1.data<scalar_t>();
			const scalar_t *G2_data = G2.data<scalar_t>();
			const scalar_t *G3_data = G3.data<scalar_t>();
			scalar_t *H_data = H.data<scalar_t>();

			for(int t=0; t<kNItems; t++) {
			forward_one_chunk_four_directions<scalar_t>
				<<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, t, kchunk, kNItems, num_, channels_, height_, width_, X_data, B_data, G1_data, G2_data, G3_data, H_data);
			C10_CUDA_CHECK(cudaGetLastError());
			}
	}));
  }
  else {
	AT_DISPATCH_REDUCED_FLOATING_TYPES(
		X.scalar_type(), "Forward_four_directions_gpu", ([&] {
			const scalar_t *X_data = X.data<scalar_t>();
			const scalar_t *B_data = B.data<scalar_t>();
			const scalar_t *G1_data = G1.data<scalar_t>();
			const scalar_t *G2_data = G2.data<scalar_t>();
			const scalar_t *G3_data = G3.data<scalar_t>();
			scalar_t *H_data = H.data<scalar_t>();

			for(int t=0; t<kNItems; t++) {
			forward_one_chunk_four_directions<scalar_t>
				<<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, t, kchunk, kNItems, num_, channels_, height_, width_, X_data, B_data, G1_data, G2_data, G3_data, H_data);
			C10_CUDA_CHECK(cudaGetLastError());
			}
	}));
  }
  return 1;
}


// bwd kernel
int Backward_four_directions(int num_, int channels_, int height_, int width_, int kNItems_,
						const at::Tensor X, 
						const at::Tensor B,
						const at::Tensor G1, const at::Tensor G2, const at::Tensor G3, 
						const at::Tensor H, 
						at::Tensor X_diff, 
						at::Tensor Bdiff,
						at::Tensor G1_diff, at::Tensor G2_diff, at::Tensor G3_diff, 
						at::Tensor H_diff)
{
  int kNItems = min(width_, kNItems_);
  int kchunk = (width_ + kNItems - 1) / kNItems;
  int count = kchunk * height_ * channels_ * num_;

  if (X.scalar_type() == at::ScalarType::Float)  {
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		X.scalar_type(), "Backward_four_directions_gpu", ([&] {
			const scalar_t *X_data = X.data<scalar_t>();
			const scalar_t *B_data = B.data<scalar_t>();
			const scalar_t *G1_data = G1.data<scalar_t>();
			const scalar_t *G2_data = G2.data<scalar_t>();
			const scalar_t *G3_data = G3.data<scalar_t>();
			const scalar_t *H_data = H.data<scalar_t>();
			scalar_t *X_diff_data = X_diff.data<scalar_t>();
			scalar_t *Bdiff_data = Bdiff.data<scalar_t>();
			scalar_t *G1_diff_data = G1_diff.data<scalar_t>();
			scalar_t *G2_diff_data = G2_diff.data<scalar_t>();
			scalar_t *G3_diff_data = G3_diff.data<scalar_t>();
			scalar_t *H_diff_data = H_diff.data<scalar_t>();

			for(int t=0; t<kNItems; t++) {
			backward_one_chunk_four_directions<scalar_t>
				<<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, t, kchunk, kNItems, num_, channels_, height_, width_, X_data, B_data, G1_data, G2_data, G3_data, H_data, X_diff_data, Bdiff_data, G1_diff_data, G2_diff_data, G3_diff_data, H_diff_data);

			C10_CUDA_CHECK(cudaGetLastError());
			}
	}));
  }
  else {
	AT_DISPATCH_REDUCED_FLOATING_TYPES(
		X.scalar_type(), "Backward_four_directions_gpu", ([&] {
			const scalar_t *X_data = X.data<scalar_t>();
			const scalar_t *B_data = B.data<scalar_t>();
			const scalar_t *G1_data = G1.data<scalar_t>();
			const scalar_t *G2_data = G2.data<scalar_t>();
			const scalar_t *G3_data = G3.data<scalar_t>();
			const scalar_t *H_data = H.data<scalar_t>();
			scalar_t *X_diff_data = X_diff.data<scalar_t>();
			scalar_t *Bdiff_data = Bdiff.data<scalar_t>();
			scalar_t *G1_diff_data = G1_diff.data<scalar_t>();
			scalar_t *G2_diff_data = G2_diff.data<scalar_t>();
			scalar_t *G3_diff_data = G3_diff.data<scalar_t>();
			scalar_t *H_diff_data = H_diff.data<scalar_t>();

			for(int t=0; t<kNItems; t++) {
			backward_one_chunk_four_directions<scalar_t>
				<<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, t, kchunk, kNItems, num_, channels_, height_, width_, X_data, B_data, G1_data, G2_data, G3_data, H_data, X_diff_data, Bdiff_data, G1_diff_data, G2_diff_data, G3_diff_data, H_diff_data);

			C10_CUDA_CHECK(cudaGetLastError());
			}
	}));
  }
  return 1;
}