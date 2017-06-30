#ifndef CAFFE_BLOCK_CIRC_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BLOCK_CIRC_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

// add by leo
#include <thrust/gather.h>                                                         
#include <thrust/transform.h>                                                      
#include <thrust/execution_policy.h>                                               
#include <thrust/sort.h>                                                           
#include <thrust/iterator/counting_iterator.h>                                     
#include <thrust/reduce.h>                                                         
#include <thrust/execution_policy.h>                                               
struct GenBlockCircIdx                                                             
{                                                                                  
    int M_, N_, K_, col_;                                                                
    GenBlockCircIdx(int M, int N, int K, int col) : 
      M_(M), N_(N), K_(K), col_(col)  {}                 
    __device__                                                                     
    int operator () (int idx)                                                      
    {                                                                              
      // get element position in A                                                 
      int r = idx / col_; // (N_*K_);                                                       
      int c = idx % col_; // (N_*K_);                                                       
                                                                                   
      // get ij-th block                                                           
      int i = r / K_;                                                              
      int j = c / K_;                                                              
                                                                                   
      // get element position within block                                         
      int p = r % K_;                                                              
      int q = c % K_;                                                              
                                                                                   
      // load value                                                                
      int idx_ = (i * N_ + j) * K_ + (K_ - q + p) % K_;                            
      return idx_;                                                                 
    }                                                                              
};                                                                                 
                                                                                   

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BlockCircInnerProductLayer : public Layer<Dtype> {
 public:
  explicit BlockCircInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BlockCircInnerProductLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  // add by leo
  int P_;
  shared_ptr<Blob<int> > idx_r_;
  shared_ptr<Blob<Dtype> > weight_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_BLOCK_CIRC_INNER_PRODUCT_LAYER_HPP_
