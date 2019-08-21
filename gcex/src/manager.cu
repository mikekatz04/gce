/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "global.h"

using namespace std;

#define NUM_THREADS 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

GCE::GCE (int phase_bins_, int mag_bins_){
    phase_bins = phase_bins_;
    mag_bins = mag_bins_;

    phase_bin_edges = new fod[2*phase_bins];
    cudaMalloc(&d_phase_bin_edges, (2*phase_bins)*sizeof(fod));

    fod bin_start = 0.0;
    fod bin_end = 2./((fod) phase_bins + 1);
    fod dbin = 1./((fod) phase_bins + 1);
    for (int i=0; i<(phase_bins); i+=1){
        phase_bin_edges[2*i] = bin_start + dbin*i;
        phase_bin_edges[2*i+1] = bin_end + dbin*i;
        //printf("%lf - %lf\n", phase_bin_edges[2*i], phase_bin_edges[2*i+1]);
    }
    phase_bin_edges[2*phase_bins-1] = 1.000001;
    phase_bin_edges[0] = -0.000001;
    cudaMemcpy(d_phase_bin_edges, phase_bin_edges, (2*phase_bins)*sizeof(fod), cudaMemcpyHostToDevice);
}


void GCE::conditional_entropy(fod *ce_vals, int num_lcs_, fod *time_vals, int *mag_bin_inds, int *num_pts_arr, int num_pts_max, fod *freqs, int num_freqs_, fod *pdots, int num_pdots_, fod *min_light_curve_times){
    num_lcs = num_lcs_;
    num_freqs = num_freqs_;
    num_pdots = num_pdots_;

    // allocate and copy
    gpuErrchk(cudaMalloc(&d_time_vals, num_lcs*num_pts_max*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_mag_bin_inds, num_lcs*2*num_pts_max*sizeof(int)));

    gpuErrchk(cudaMemcpy(d_time_vals, time_vals, num_lcs*num_pts_max*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mag_bin_inds, mag_bin_inds, num_lcs*2*num_pts_max*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_freqs, num_freqs*sizeof(fod)));
    gpuErrchk(cudaMemcpy(d_freqs, freqs, num_freqs*sizeof(fod), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_pdots, num_pdots*sizeof(fod)));
    gpuErrchk(cudaMemcpy(d_pdots, pdots, num_pdots*sizeof(fod), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_ce_vals, num_pdots*num_freqs*num_lcs*sizeof(fod)));

    gpuErrchk(cudaMalloc(&d_num_pts_arr, num_lcs*sizeof(int)));
    gpuErrchk(cudaMemcpy(d_num_pts_arr, num_pts_arr, num_lcs*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_min_light_curve_times, num_lcs*sizeof(fod)));
    gpuErrchk(cudaMemcpy(d_min_light_curve_times, min_light_curve_times, num_lcs*sizeof(fod), cudaMemcpyHostToDevice));


    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    dim3 griddim(num_lcs, nblocks, num_pdots);
    kernel<<<griddim, NUM_THREADS, 2*phase_bins*sizeof(fod)>>>(d_ce_vals, d_freqs, num_freqs, d_pdots, num_pdots, d_phase_bin_edges, d_mag_bin_inds, d_time_vals, d_num_pts_arr, num_pts_max, mag_bins, phase_bins, num_lcs, d_min_light_curve_times);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // transfer ce vals
    cudaMemcpy(ce_vals, d_ce_vals, num_pdots*num_freqs*num_lcs*sizeof(fod), cudaMemcpyDeviceToHost);

    // free all the memory
    gpuErrchk(cudaFree(d_time_vals));
    gpuErrchk(cudaFree(d_mag_bin_inds));
    gpuErrchk(cudaFree(d_freqs));
    gpuErrchk(cudaFree(d_pdots));
    gpuErrchk(cudaFree(d_ce_vals));
    gpuErrchk(cudaFree(d_num_pts_arr));
    gpuErrchk(cudaFree(d_min_light_curve_times));
}

GCE::~GCE() {
    delete[] phase_bin_edges;
    cudaFree(d_phase_bin_edges);
}
