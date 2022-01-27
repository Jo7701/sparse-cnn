#ifndef _HYPERPARAMS
#define _HYPERPARAMS

static const int tile_ichan = 8;
static const int tile_ochan = 8;
static const int tile_ofm_width = 4;
static const int tile_ofm_height = 2;
static const int padding = 1;
static const int tile_ifm_width = tile_ofm_width+2*padding;
static const int tile_ifm_height = tile_ofm_height+2*padding;

#endif
