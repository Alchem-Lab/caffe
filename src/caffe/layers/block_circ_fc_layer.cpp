#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/block_circ_fc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.block_circ_fc_param().num_output();
  bias_term_ = this->layer_param_.block_circ_fc_param().bias_term();
  transpose_ = this->layer_param_.block_circ_fc_param().transpose();
  // add by leo
  P_ = this->layer_param_.block_circ_fc_param().partition_size();

  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.block_circ_fc_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    
    // add by leo
    //CHECK_EQ(0, K_ % P_) << "Leo: K_ % P_ != 0";
    //CHECK_EQ(0, N_ % P_) << "Leo: N_ % P_ != 0";
    //LOG(ERROR) << "Leo: N_=" << N_ << ", K_=" << K_ << ", P_=" << P_;
    //this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		
		const int num_weights = weight_shape[0] * weight_shape[1];
		weight_buffer_.reset(new Blob<Dtype>(weight_shape));
    idx_r_.reset(new Blob<int>(weight_shape));

    this->blobs_[0].reset(new Blob<Dtype>(num_weights / P_, 1, 1, 1));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.block_circ_fc_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

 
    // fill the weights
    //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    //    this->layer_param_.inner_product_param().weight_filler()));
    //weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.block_circ_fc_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // add by leo
  // this can be a little dangerous. 
  // I am not sure if this is the right time to clear diff considering following facts:
  // 1. Backward gemm set the beta as "1." rather than "0.", looks like accumulating
  // 2. Solver.cpp calls "net_->ClearParamDiffs();" at the begining of each iter
  // ( following code are from "ClearParamDiffs()" )
	switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(weight_buffer_->count(), static_cast<Dtype>(0),
                weight_buffer_->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(weight_buffer_->count(), static_cast<Dtype>(0),
                    weight_buffer_->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
  }
  //LOG(INFO) << "Leo: BlockCircInnerProductLayer::Reshape()";
  /*
Iteration 9991, lr = 0.000594804
I0217 11:58:09.815868 22331 block_circ_fc_layer.cpp:99] Leo: BlockCircInnerProductLayer::Reshape()
I0217 11:58:09.815878 22331 block_circ_fc_layer.cu:16] Leo: BlockCircInnerProductLayer::Forward_gpu()
I0217 11:58:09.816319 22331 block_circ_fc_layer.cu:57] Leo: BlockCircInnerProductLayer::Backward_gpu()
I0217 11:58:09.817769 22331 solver.cpp:228] Iteration 9992, loss = 0.00585739
   */


  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(0, 1) << "Leo: BlockCircInnerProductLayer doesn't support cpu";
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  //Dtype* top_data = top[0]->mutable_cpu_data();

  //// add by leo
  //LOG(INFO) << "Leo: BlockCircInnerProductLayer::Forward_gpu()";
	//thrust::transform(
  //    thrust::host,
  //		thrust::make_counting_iterator(0),
  // 		thrust::make_counting_iterator(weight_buffer_->count()),
  // 		idx_r_->mutable_cpu_data(),
  // 		GenBlockCircIdx(weight_buffer_->shape(0) / P_, weight_buffer_->shape(1) / P_, P_));
  //thrust::gather(
  //   thrust::host,
  //   idx_r_->cpu_data(),
  //   idx_r_->cpu_data() + idx_r_->count(),
  //   this->blobs_[0]->cpu_data(),
  //   weight_buffer_->mutable_cpu_data()); 

  ////const Dtype* weight = this->blobs_[0]->cpu_data();
  //const Dtype* weight = this->weight_buffer_->cpu_data();

  //caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
  //    M_, N_, K_, (Dtype)1.,
  //    bottom_data, weight, (Dtype)0., top_data);
  //if (bias_term_) {
  //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
  //      bias_multiplier_.cpu_data(),
  //      this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  //}
}

template <typename Dtype>
void BlockCircInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(0, 1) << "Leo: BlockCircInnerProductLayer doesn't support cpu";
}

#ifdef CPU_ONLY
STUB_GPU(BlockCircInnerProductLayer);
#endif

INSTANTIATE_CLASS(BlockCircInnerProductLayer);
REGISTER_LAYER_CLASS(BlockCircInnerProduct);

}  // namespace caffe
