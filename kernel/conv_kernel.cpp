#include "hyperparams.h"
#include <string.h>

extern "C" {
    void compute_engine(int ofm_buffer[tile_ochan][tile_ofm_height][tile_ofm_width], int ifm_buffer[tile_ichan][tile_ifm_height][tile_ifm_width],
                        int kernel_buffer[tile_ochan][tile_ichan][9],
                        int row_idx_buffer[tile_ochan][tile_ichan][9],
                        int col_idx_buffer[tile_ochan][tile_ichan][9], int num_nonzero) {
        int i,j;
        //COMPUTE ENGINE
        COMPUTE_OFM_HEIGHT: for(int ty = 0; ty < tile_ofm_height; ty++){
            COMPUTE_OFM_WIDTH: for(int tx = 0; tx < tile_ofm_width; tx++){
                COMPUTE_WEIGHT_ITER: for(int iter = 0; iter < num_nonzero; iter++){
                    COMPUTE_OCHAN: for(int to = 0; to < tile_ochan; to++){
                        #pragma hls unroll
                        COMPUTE_ICHAN: for(int ti = 0; ti < tile_ichan; ti++){
                            #pragma hls unroll

                            i = row_idx_buffer[to][ti][iter];
                            j = col_idx_buffer[to][ti][iter];

                            // Calculate input padding boundaries
                            int xVal = tx + j, yVal = ty + i;
                            ofm_buffer[to][ty][tx] += ifm_buffer[ti][yVal][xVal] * kernel_buffer[to][ti][iter];
                        }
                    }
                }
            }
        }
    }
}

extern "C" {
    void load_tile_ifm(int* image, int ifm_buffer[tile_ichan][tile_ifm_height][tile_ifm_width],
                       int input, int x, int y, int i_size, int i_chan){

    	ZERO_IFM_ICHAN: for(int ti = 0; ti < tile_ichan; ti++){
			#pragma hls unroll
			ZERO_IFM_HEIGHT: for(int ty = 0; ty < tile_ifm_height; ty++){
				#pragma hls unroll
				ZERO_IFM_WIDTH: for(int tx = 0; tx < tile_ifm_width; tx++){
					#pragma hls unroll
						ifm_buffer[ti][ty][tx] = 0;
				}
			}
		}

    	int ichan_bound = tile_ichan;
    	if(input+tile_ichan > i_chan)
    		ichan_bound = i_chan-input;

		int ty_start = 0;
		if(ty_start+y-padding<0)
			ty_start = padding-y;
		int ifm_height_bound = tile_ifm_height;
		if(y+ifm_height_bound-padding > i_size)
			ifm_height_bound = i_size+padding-y;

		int tx_start = 0;
		if(tx_start+x-padding<0)
			tx_start = padding-x;
		int ifm_width_bound = tile_ifm_width;
		if(x+ifm_width_bound-padding > i_size)
			ifm_width_bound = i_size+padding-x;

        LOAD_WEIGHT_OCHAN: for(int ti = 0; ti < ichan_bound; ti++){
            LOAD_WEIGHT_ICHAN: for(int ty = ty_start; ty < ifm_height_bound; ty++){
                LOAD_WEIGHT_ITER: for(int tx = tx_start; tx < ifm_width_bound; tx++){
                    #pragma hls pipeline II=1
					int tii = ti + input;
					int tyy = ty + y - padding;
					int txx = tx + x - padding;
					ifm_buffer[ti][ty][tx] = image[tii * (i_size * i_size) + tyy * i_size + txx];
                }
            }
        }
    }
}

extern "C" {
    void load_tile_kernel(int *weight, int *row_idx, int *col_idx, int output, int input, int i_chan, int o_chan, int num_nonzero,
                          int kernel_buffer[tile_ochan][tile_ichan][9],
                          int row_idx_buffer[tile_ochan][tile_ichan][9],
                          int col_idx_buffer[tile_ochan][tile_ichan][9]){

    	ZERO_WEIGHT_OCHAN: for(int to = 0; to < tile_ochan; to++){
			#pragma hls unroll
			ZERD_WEIGHT_ICHAN: for(int ti = 0; ti < tile_ichan; ti++){
				#pragma hls unroll
				ZERD_WEIGHT_ITER: for(int iter = 0; iter < 9; iter++){
					#pragma hls unroll
					kernel_buffer[to][ti][iter] = 0;
					row_idx_buffer[to][ti][iter] = 0;
					col_idx_buffer[to][ti][iter] = 0;
				}
			}
    	}

    	int ochan_bound = tile_ochan;
    	if(output+tile_ochan > o_chan)
    		ochan_bound = o_chan-output;

    	int ichan_bound = tile_ichan;
    	if(input+tile_ichan > i_chan)
    		ichan_bound = i_chan-input;

        LOAD_WEIGHT_OCHAN: for(int to = 0; to < ochan_bound; to++){
            LOAD_WEIGHT_ICHAN: for(int ti = 0; ti < ichan_bound*num_nonzero; ti++){
				#pragma hls pipeline II=1
				int too = to + output;
				int tii = ti + input*num_nonzero;
				int iter = ti%num_nonzero;
				kernel_buffer[to][ti/num_nonzero][iter] = weight[too * i_chan * num_nonzero + tii];
				row_idx_buffer[to][ti/num_nonzero][iter] = row_idx[too * i_chan * num_nonzero + tii];
				col_idx_buffer[to][ti/num_nonzero][iter] = col_idx[too * i_chan * num_nonzero + tii];
            }
        }
    }
}

extern "C" {
    void store_tile_ofm(int *out, int ofm_buffer[tile_ochan][tile_ofm_height][tile_ofm_width], int output, int o_chan, int o_size, int x, int y){
        int ochan_bounds = tile_ochan;
        if(output+ochan_bounds > o_chan)
        		ochan_bounds = o_chan-output;

        int height_bounds = tile_ofm_height;
        if(y+height_bounds > o_size)
        		height_bounds = o_size-y;

        int width_bounds = tile_ofm_width;
        if(x+width_bounds > o_size)
        		width_bounds = o_size-x;

    	STORE_OFM_OCHAN: for(int to = 0; to < ochan_bounds; to++){
            STORE_OFM_HEIGHT: for(int ty = 0; ty < height_bounds; ty++){
                STORE_OFM_WIDTH: for(int tx = 0; tx < width_bounds; tx++){
                    #pragma hls pipeline II=1
                    int too = to + output;
                    int tyy = ty + y;
                    int txx = tx + x;

                    out[(too*o_size+tyy)*o_size + txx] = ofm_buffer[to][ty][tx];
                }
            }
        }
    }
}


extern "C" {
    void sparse_conv(
        int *weight,
        int *row_idx,
        int *col_idx,
        int *image,
        int *out,
        int i_chan,
        int o_chan,
        int num_nonzero,
        int i_size,
        int o_size,
        int stride,
        int padding,
        int relu_on
    )
    {

        #pragma HLS INTERFACE m_axi port=row_idx offset=slave bundle=gmem0
        #pragma HLS INTERFACE m_axi port=col_idx offset=slave bundle=gmem1
        #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2
        #pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem3
        #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem3

        const int zero[tile_ochan*tile_ofm_height*tile_ofm_width] = {0};

        int ifm_buffer[tile_ichan][tile_ifm_height][tile_ifm_width];
        int kernel_buffer[tile_ochan][tile_ichan][9];
        int row_idx_buffer[tile_ochan][tile_ichan][9];
        int col_idx_buffer[tile_ochan][tile_ichan][9];
        int ofm_buffer[tile_ochan][tile_ofm_height][tile_ofm_width];

        #pragma hls array_partition variable=ifm_buffer complete dim=0
        #pragma hls array_partition variable=kernel_buffer complete dim=0
        #pragma hls array_partition variable=row_idx_buffer complete dim=0
        #pragma hls array_partition variable=col_idx_buffer complete dim=0
        #pragma hls array_partition variable=ofm_buffer complete dim=0

        // Runs over output pixel in Y-direction
        MAIN_OFM_HEIGHT: for(int y = 0; y < o_size; y+=tile_ofm_height){
            // Runs over output pixel in X-direction
            MAIN_OFM_WIDTH: for(int x = 0; x < o_size; x+=tile_ofm_width){
                // Runs over output filters
                MAIN_OCHAN: for(int output = 0; output < o_chan; output+=tile_ochan){
                    memcpy(ofm_buffer, zero, sizeof(ofm_buffer));
                    // Runs over each input channel of input feature map
                    MAIN_ICHAN: for(int input = 0; input < i_chan; input+=tile_ichan){
                        load_tile_ifm(image, ifm_buffer, input, x, y, i_size, i_chan);
                        load_tile_kernel(weight, row_idx, col_idx, output, input, i_chan, o_chan, num_nonzero,
                                         kernel_buffer, row_idx_buffer, col_idx_buffer);
                        compute_engine(ofm_buffer, ifm_buffer, kernel_buffer, row_idx_buffer, col_idx_buffer, num_nonzero);
                    }
                    store_tile_ofm(out, ofm_buffer, output, o_chan, o_size, x, y);
                }
            }
        }
    }
}
