#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/block_circ_fc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // add by leo
  //LOG(INFO) << "Leo: BlockCircInnerProductLayer::Forward_gpu()";
	thrust::transform(
      thrust::device,
  		thrust::make_counting_iterator(0),
   		thrust::make_counting_iterator(weight_buffer_->count()),
   		idx_r_->mutable_gpu_data(),
   		GenBlockCircIdx(weight_buffer_->shape(0) / P_, 
                      weight_buffer_->shape(1) / P_, 
                      P_,
                      weight_buffer_->shape(1)));
  thrust::gather(
     thrust::device,
     idx_r_->gpu_data(),
     idx_r_->gpu_data() + idx_r_->count(),
     this->blobs_[0]->gpu_data(),
     weight_buffer_->mutable_gpu_data()); 

  const Dtype* weight = this->weight_buffer_->gpu_data();
  
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "Leo: BlockCircInnerProductLayer::Backward_gpu()";
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // add by leo
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->weight_buffer_->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->weight_buffer_->mutable_gpu_diff());
    }
    thrust::stable_sort_by_key(
        thrust::device,
        idx_r_->mutable_gpu_data(),
        idx_r_->mutable_gpu_data() + idx_r_->count(),
        this->weight_buffer_->mutable_gpu_diff());
    thrust::reduce_by_key(
        thrust::device,
        idx_r_->gpu_data(),
        idx_r_->gpu_data() + idx_r_->count(),
        this->weight_buffer_->mutable_gpu_diff(),
        thrust::make_discard_iterator(),
        this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    // add by leo
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight_buffer_->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight_buffer_->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
    //if (transpose_) {
    //  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    //      M_, K_, N_,
    //      (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
    //      (Dtype)0., bottom[0]->mutable_gpu_diff());
    //} else {
    //  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    //      M_, K_, N_,
    //     (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
    //     (Dtype)0., bottom[0]->mutable_gpu_diff());
    //}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BlockCircInnerProductLayer);

}  // namespace caffe
