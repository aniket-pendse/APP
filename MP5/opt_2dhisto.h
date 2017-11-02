#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(size_t height, size_t width);
/* Include below the function headers of any other functions that you implement */

void preallocate_memory(uint32_t *input[],size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH], uint32_t * temp_input);

void deallocate_memory(uint32_t * host_bins, uint32_t * device_bin, size_t histo_height, size_t histo_width);

extern uint32_t *d_input;
extern uint32_t *device_bins;
#endif
