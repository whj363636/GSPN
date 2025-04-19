// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

// Forward Launcher
int Forward_four_directions(int num_, int channels_, int height_, int width_, int kNItems_, 
						const at::Tensor X, 
						const at::Tensor B, 
						const at::Tensor G1, const at::Tensor G2, const at::Tensor G3, 
						const at::Tensor H);

// Backward Launcher
int Backward_four_directions(int num_, int channels_, int height_, int width_, int kNItems_,
						const at::Tensor X, 
						const at::Tensor B, 
						const at::Tensor G1, const at::Tensor G2, const at::Tensor G3, 
						const at::Tensor H, 
						at::Tensor X_diff, 
						at::Tensor B_diff,
						at::Tensor G1_diff, at::Tensor G2_diff, at::Tensor G3_diff, 
						at::Tensor H_diff);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int gaterecurrent2dnoind_forward_cuda(int kNItems_,
									at::Tensor X, 
									at::Tensor B,  
									at::Tensor G1, 
									at::Tensor G2, 
									at::Tensor G3, 
									at::Tensor output)
{
	CHECK_INPUT(X);
	CHECK_INPUT(B);
	CHECK_INPUT(G1);
	CHECK_INPUT(G2);
	CHECK_INPUT(G3);
	CHECK_INPUT(output);

	// dimensions
	int num_ = X.size(0);
	int channels_ = X.size(1);
	int height_ = X.size(2);
	int width_ = X.size(3);

	//const int count = height_ * channels_ * num_;
	Forward_four_directions(num_, channels_, height_, width_, kNItems_, X, B, G1, G2, G3, output);

	return 1;
}


int gaterecurrent2dnoind_backward_cuda(int kNItems_,
										at::Tensor top, 
										at::Tensor top_grad, 
										at::Tensor X, 
										at::Tensor B,  
										at::Tensor G1, 
										at::Tensor G2, 
										at::Tensor G3, 
										at::Tensor X_diff, 
										at::Tensor B_diff,
										at::Tensor G1_diff, 
										at::Tensor G2_diff, 
										at::Tensor G3_diff)
{
	CHECK_INPUT(top);
	CHECK_INPUT(top_grad);
	CHECK_INPUT(X);
	CHECK_INPUT(B);
	CHECK_INPUT(G1);
	CHECK_INPUT(G2);
	CHECK_INPUT(G3);
	CHECK_INPUT(X_diff);
	CHECK_INPUT(B_diff);
	CHECK_INPUT(G1_diff);
	CHECK_INPUT(G2_diff);
	CHECK_INPUT(G3_diff);

	// dimensions
	int num_ = X.size(0);
	int channels_ = X.size(1);
	int height_ = X.size(2);
	int width_ = X.size(3);

	Backward_four_directions(num_, channels_, height_, width_, kNItems_,
						X, 
						B, 
						G1, G2, G3, 
						top, 
						X_diff, 
						B_diff, 
						G1_diff, G2_diff, G3_diff, 
						top_grad);
	return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gaterecurrent2dnoind_forward_cuda, "gaterecurrent2dnoind forward (CUDA)");
  m.def("backward", &gaterecurrent2dnoind_backward_cuda, "gaterecurrent2dnoind backward (CUDA)");
}