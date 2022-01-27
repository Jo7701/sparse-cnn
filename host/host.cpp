#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <chrono>
#include <CL/cl2.hpp>
#include <CL/cl_ext.h>
#include <stdio.h>
#include "stdint.h"
#include <stdlib.h>
#include <hls_half.h>
typedef float datatype_inh;
//#include "ap_fixed.h"
//typedef ap_fixed<8,8> datatype_inh;
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <fcntl.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx"){
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if(access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    return buf;
}

int load_sparse_weights(
        int *weights,
        int *row_idx,
        int *col_idx,
        int i_chan,
            int o_chan,
            int w_size,
        FILE *fid)
{

  //shape: (o_chan, i_chan, w_size, w_size)

  int  j;
  int index = 0;
  float data;
  char str[300];
  for (j = 0; j < i_chan * o_chan * w_size * w_size; j++)
  {

    fscanf(fid,"%s", str);
    data = atoi(str);
    if(data != 0){
      weights[index] = (int) data;
      row_idx[index] = (j / w_size) % w_size;
      col_idx[index] = j % w_size;
      index++;
    }
    fscanf(fid,"\n");
  }

  return index;
}

void sparse_conv(
  int *weight,
  int *row_idx,
  int *col_idx,
  int *image,
  int *out,
  int num_nonzero,
    int i_chan,
    int o_chan,
    int i_size,
    int o_size,
    int w_size,
    int stride,
    int padding,
    int relu_on
)
{
  int i,j, curr_index;

    // Runs over output filters
    for(int output = 0; output < o_chan; output++){
        // Runs over output pixel in Y-direction
        for(int y = 0; y < o_size; y++){
            // Runs over output pixel in X-direction
            for(int x = 0; x < o_size; x++){
              int acc = 0;
                // Runs over each input channel of input feature map
                for(int input = 0; input < i_chan; input++){
                    // Runs over nonzero elems in filter window
                    for(int iter = 0; iter < num_nonzero; iter++){
                      curr_index = output * i_chan * num_nonzero + num_nonzero * input + iter;
                      i = row_idx[curr_index];
                      j = col_idx[curr_index];

                    // Calculate input padding boundaries
                        int xVal = x*stride + j-padding, yVal = y*stride + i-padding;

                        // Convolution operation
                        if(yVal >= 0 && yVal < i_size && xVal >= 0 && xVal < i_size){
                            acc += (int) image[(input*i_size + yVal)*i_size + xVal] *
                                   (int) weight[curr_index];
                        }
                    }
        }

        if(relu_on == 1 && acc < 0)
              //    std::cout << "result " << acc << std::endl;
                  out[(output*o_size + y)*o_size + x] = 0;

        else
          out[(output*o_size + y)*o_size + x] = acc;

          }
        }
    }
}

void verify(int* expected, int* output, int size) {
  for(int i = 0; i < size; i++){
    if(expected[i] != output[i]) {
      printf("Mismatch at %d. Expected %d. Got %d\n", i, expected[i], output[i]);
      exit(-1);
    }
  }
}


//int main()
//{
int main(int argc, char** argv)
{
//init opencl environment
  cl_int err;
  std::string binaryFile = (argc != 2) ? "binary_container_1.xclbin" : argv[1];
  unsigned fileBufSize;
  std::vector<cl::Device> devices = get_xilinx_devices();
  cl::Device device = devices[0];
  cl::Context context(device, NULL, NULL, NULL, &err);
  char* fileBuf = read_binary_file(binaryFile, fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  cl::Program program(context, {device}, bins, NULL, &err); //program the device
  std::cout << err << std::endl;
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err); //setup command queue that will take in diff commands

  cl::Kernel kernel_conv(program, "sparse_conv", &err);
  std::cout << "HERE" << std::endl;
  int num_inp_channels = 8;
  int num_out_channels = 8;
  int num_nonzero = 3;
  int k_size = 3;
  int img_size = 224;
  int total_kernel_bytes = num_inp_channels * num_out_channels * num_nonzero * sizeof(int);
  int total_row_bytes = num_inp_channels * num_out_channels * num_nonzero * sizeof(int);
  int total_col_bytes = num_inp_channels * num_out_channels * num_nonzero * sizeof(int);
  int total_img_bytes = num_inp_channels * img_size * img_size * sizeof(int);
  int total_output_bytes = num_out_channels * img_size * img_size * sizeof(int);

  std::cout << "Kernel bytes: " << total_kernel_bytes << std::endl;
  std::cout << "Row bytes: " << total_row_bytes << std::endl;
  std::cout << "Col bytes: " << total_col_bytes << std::endl;
  std::cout << "Input bytes: " << total_img_bytes << std::endl;
  std::cout << "Output bytes: " << total_output_bytes << std::endl;
  std::cout << "Total bytes: " << total_kernel_bytes + total_row_bytes + total_col_bytes + total_img_bytes + total_output_bytes << std::endl;

  cl_mem_ext_ptr_t rowExt, colExt, weightExt, inExt, outExt;  // Declaring extensions for buffers
  rowExt.flags  = XCL_MEM_DDR_BANK0; // Specify Bank0 Memory for row_index memory
  colExt.flags  = XCL_MEM_DDR_BANK1;
  weightExt.flags  = XCL_MEM_DDR_BANK2;
  inExt.flags  = XCL_MEM_DDR_BANK3;
  outExt.flags = XCL_MEM_DDR_BANK3; // Specify Bank1 Memory for output Memory
  inExt.obj = 0; rowExt.obj = 0; colExt.obj = 0; weightExt.obj = 0; outExt.obj = 0; // Setting Obj and Param to Zero
  inExt.param = 0 ; rowExt.param = 0; colExt.param = 0; weightExt.param = 0; outExt.param = 0;



  cl::Buffer row_buf(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, total_row_bytes, &rowExt, &err);
  if(err < 0){
      std::cout << "ERROR2: " << err << std::endl;
      return -1;
    }
  cl::Buffer col_buf(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, total_col_bytes, &colExt, &err);
  if(err < 0){
      std::cout << "ERROR3: " << err << std::endl;
      return -1;
    }
  cl::Buffer inp_buf(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, total_img_bytes, &inExt, &err);
  if(err < 0){
      std::cout << "ERROR4: " << err << std::endl;
      return -1;
    }
  cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, total_output_bytes, &outExt, &err);
  if(err < 0){
      std::cout << "ERROR5: " << err << std::endl;
      return -1;
    }
  cl::Buffer kernel_buf(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, total_kernel_bytes, &weightExt, &err);
    if(err < 0){
      std::cout << "ERROR1: " << err << std::endl;
      return -1;
    }

  int* weights = (int*) q.enqueueMapBuffer(kernel_buf, CL_TRUE, CL_MAP_WRITE, 0, total_kernel_bytes);
  int* row_idx = (int*) q.enqueueMapBuffer(row_buf, CL_TRUE, CL_MAP_WRITE, 0, total_row_bytes);
  int* col_idx = (int*) q.enqueueMapBuffer(col_buf, CL_TRUE, CL_MAP_WRITE, 0, total_col_bytes);
  int* image = (int*) q.enqueueMapBuffer(inp_buf, CL_TRUE, CL_MAP_WRITE, 0, total_img_bytes);
  int* output = (int*) q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, total_output_bytes);

  if(weights == NULL || row_idx == NULL || col_idx == NULL || image == NULL || output == NULL){
	  std::cout << "ERROR: NULL POINTER" << std::endl;
	  return -1;
  }

  char buf[100];
  FILE* fid;
  fid = fopen("/home/jrejive2/workspace/sparse_cnn/src/image.txt", "r");
  if(fid == NULL) {
	  std::cout << "ERROR" << std::endl;
	  return -1;
  }

  for(int i = 0; i < 3 * img_size * img_size; i++){
    fscanf(fid, "%s", buf);
    image[i] = std::stoi(buf);
    fscanf(fid, "\n");
  }

  fclose(fid);

  fid = fopen("/home/jrejive2/workspace/sparse_cnn/src/weights.txt", "r");
  if(fid == NULL) {
  	  std::cout << "ERROR" << std::endl;
  	  return -1;
    }
  load_sparse_weights(weights, row_idx, col_idx, num_inp_channels, num_out_channels, k_size, fid);


  kernel_conv.setArg(0, kernel_buf);
  kernel_conv.setArg(1, row_buf);
  kernel_conv.setArg(2, col_buf);
  kernel_conv.setArg(3, inp_buf);
  kernel_conv.setArg(4, out_buf);
  kernel_conv.setArg(5, num_inp_channels);
  kernel_conv.setArg(6, num_out_channels);
  kernel_conv.setArg(7, num_nonzero);
  kernel_conv.setArg(8, img_size);
  kernel_conv.setArg(9, img_size);
  kernel_conv.setArg(10, 1);
  kernel_conv.setArg(11, 1);
  kernel_conv.setArg(12, 0);


  cl::Event event;
  uint64_t start, end;

  q.enqueueMigrateMemObjects({kernel_buf, row_buf, col_buf, inp_buf}, 0);
  q.enqueueTask(kernel_conv, NULL, &event);
  q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();

  err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &start);
  err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &end);

  int* expected = new int[num_out_channels * img_size * img_size];
  for(int i = 0; i < num_out_channels * img_size * img_size; i++)
      expected[i] = 0;

  sparse_conv(weights, row_idx, col_idx, image, expected, num_nonzero, num_inp_channels, num_out_channels, img_size, img_size, k_size, 1, 1, 0);
  verify(expected, output, num_out_channels * img_size * img_size);

  auto conv_time = end - start;
  std::cout << "TEST PASSED" << std::endl;
  std::cout << "Conv Time: " << conv_time / 1000000 << "ms" << std::endl;

//  clReleaseContext(context());

  delete[] fileBuf;
  delete[] expected;

  return EXIT_SUCCESS;
}
