#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "float32.h"

#define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
#define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

int main() {
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnPoolingDescriptor_t pooling_desc;
  //create descriptor handle
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  //initialize descriptor
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         CUDNN_POOLING_MAX,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         3,                       //window height
                                         3,                       //window width
                                         0,                       //vertical padding
                                         0,                       //horizontal padding
                                         1,                       //vertical stride
                                         1));                     //horizontal stride
  
  cudnnTensorDescriptor_t in_desc;
  //create input data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
  //initialize input data descriptor 
  checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,                  //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        1,                        //number of images
                                        1,                        //number of channels
                                        5,                       //data height 
                                        5));                     //data width

  cudnnTensorDescriptor_t out_desc;
  //create output data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
  //initialize output data descriptor
  checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,                 //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        1,                        //number of images
                                        1,                        //number of channels
                                        3,                        //data height
                                        3));                      //data width

  stype alpha = 1.0f;
  stype beta = 0.0f;
  //GPU data pointers
  dtype *in_data, *out_data;
  //allocate arrays on GPU
  cudaMalloc(&in_data,IN_DATA_BYTES);
  cudaMalloc(&out_data,OUT_DATA_BYTES);
  //copy input data to GPU array
  cudaMemcpy(in_data,input,IN_DATA_BYTES,cudaMemcpyHostToDevice);
  //initize output data on GPU
  cudaMemset(out_data,0,OUT_DATA_BYTES);

  //Call pooling operator
  checkCUDNN(cudnnPoolingForward(cudnn,         //cuDNN context handle
                                 pooling_desc,  //pooling descriptor handle
                                 &alpha,        //alpha scaling factor
                                 in_desc,       //input tensor descriptor
                                 in_data,       //input data pointer to GPU memory
                                 &beta,         //beta scaling factor
                                 out_desc,      //output tensor descriptor
                                 out_data));    //output data pointer from GPU memory

  //allocate array on CPU for output tensor data
  dtype *result = (dtype*)malloc(OUT_DATA_BYTES);
  //copy output data from GPU
  cudaMemcpy(result,out_data,OUT_DATA_BYTES,cudaMemcpyDeviceToHost);

  //loop over and check that the forward pass outputs match expected results (exactly)
  int err = 0;
  for(int i=0; i<OUT_SIZE; i++) {
    if(result[i] != output[i]) {
      std::cout << "Error! Expected " << output[i] << " got " << result[i] << " for idx " << i <<std::endl;
      err++;
    }
  }

  std::cout << "Forward finished with " << err << " errors" << std::endl;

  for(int i=0; i<3; i++)
  {
    for(int j=0; j<3; j++)
    {
      printf("%.2f ", result[i * 3 + j]);
    }
    printf("\n");
  }

  //free CPU arrays
  free(result);

  //free GPU arrays
  cudaFree(in_data);
  cudaFree(out_data);

  //free cuDNN descriptors
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  cudnnDestroy(cudnn);
  
  return 0;
}