// Copyright (C) Michael Katz (2020)
//
// This file is part of gce
//
// gce is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// gce is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with gce.  If not, see <http://www.gnu.org/licenses/>


#include <math.h>
#include <random>
#include "global.h"


#define NUM_THREADS 64


// function for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Custom mod function
__device__
fod mod_func(fod x, fod y)
{
    int num = (int) (x/y);

    fod check = x - (num*y);

    return check;
}


// Phase fold of time and pick phase bins
__device__
int get_phase_bin(fod t_val, fod pdot, fod frequency, fod period, fod half_dbins, int phase_bins){
    fod folded_val = mod_func((fod) t_val-0.5*pdot*frequency*(t_val*t_val), (fod)period)*((fod)frequency); // between 0 and 1

    // revert negative fold values
    if (folded_val < 0.0) folded_val *= -1.0;

    // find phase bin
    int j = (int) abs((folded_val/(fod)half_dbins));
    j = j % phase_bins;

    return j;
}


__device__ fod ce (fod frequency, fod pdot,
                   int* __restrict__ mag_bin_inds, fod* __restrict__ time_vals, int npoints,
                   int mag_bins, int phase_bins, int lc_i,
                   fod * temp_phase_prob, fod *overall_phase_prob, fod half_dbins){

    fod period = 1./frequency;
    fod folded_val = 0.0;

    fod sum_ij = 0.0;
    fod t_val = 0.0;
    fod total_points = 0.0;
    int j = 0;
    int j1 = 0;

    // clear phase values
    for (int jj = 0; jj<phase_bins; jj+=1){
            overall_phase_prob[jj] = 0.0;
            temp_phase_prob[jj] = 0.0;
        }

    int ind_mag=0;
    int temp_ind = 0;
    int gap = 0;

    // get population of overall phase bins
    for (int k=0; k<npoints; k++){

            t_val = time_vals[k];
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            overall_phase_prob[j] += 1.0;
            total_points += 1.0;

            // account for 50% overlap
            int j1 = (j <= 0) ? j = phase_bins - 1 : (j - 1) % phase_bins;
            overall_phase_prob[j1] += 1.0;
            total_points += 1.0;

        }

    // now find population of each individual bin
    // operations are done within a 1d histogam of phase bins (single mag)

    ind_mag = mag_bin_inds[0];
    int kk = 0;
    while (kk < npoints){
        while ((mag_bin_inds[kk] == ind_mag)){
            t_val = time_vals[kk];
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            // fill singular bin

            temp_phase_prob[j] += 1.0;
            int j1 = (j <= 0) ? j = phase_bins - 1 : (j - 1) % phase_bins ;
            temp_phase_prob[j1] += 1.0;

            kk += 1;
            if (kk == npoints) break;
        }

        // sum all bins at this magnitude
        for (int jj = 0; jj<phase_bins; jj+=1){
            if ((overall_phase_prob[jj] > 0.0) && (temp_phase_prob[jj] > 0.0)){
                sum_ij += temp_phase_prob[jj]*log(overall_phase_prob[jj]/temp_phase_prob[jj]);
                temp_phase_prob[jj] = 0.0;
            }
        }
        if (kk < npoints) ind_mag = mag_bin_inds[kk];
    }

    return  sum_ij/((fod) total_points);
}


// Function wrap for conditional entropy
__global__ void kernel(fod* __restrict__ ce_vals, fod* __restrict__ freqs, int num_freqs, fod* __restrict__ pdots,
                       int num_pdots, int* __restrict__ mag_bin_inds,
                       fod* __restrict__ time_vals, int * __restrict__ num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs, fod half_dbins){

    // declare and assign dynamic shared memory
    extern __shared__ fod time_vals_share[];
    int *mag_bin_inds_share = (int*)(&time_vals_share[num_pts_max]);
    fod * overall_phase_prob = (fod*) &time_vals_share[2*num_pts_max];
    fod *temp_phase_prob = (fod*) &time_vals_share[2*num_pts_max + phase_bins*NUM_THREADS];

    __syncthreads();


    for (int lc_i = blockIdx.y;
         lc_i < num_lcs;
         lc_i += gridDim.y) {

     int num_pts_this_lc = num_pts_arr[lc_i];

     // use coalesced memory transfers to load time and magnitude values
     for (int j=threadIdx.x; j<num_pts_this_lc; j+=blockDim.x){

         time_vals_share[j] = time_vals[lc_i*num_pts_max + j];
         mag_bin_inds_share[j] = mag_bin_inds[lc_i*num_pts_max + j];

     }

     __syncthreads();

   for (int pdot_i = blockIdx.z;
        pdot_i < num_pdots;
        pdot_i += gridDim.z) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_freqs;
         i += blockDim.x * gridDim.x) {

    // run ce computation for this set of parameters
    ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = ce(freqs[i], pdots[pdot_i], mag_bin_inds_share,
                                                          &time_vals_share[0], num_pts_this_lc, mag_bins, phase_bins,
                                                          lc_i,
                                                          &temp_phase_prob[threadIdx.x*phase_bins], &overall_phase_prob[threadIdx.x*phase_bins], half_dbins);
  }
}
}
}


// wrapper function callable from another script or cython
void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
             int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins)
{

    // number of blocks given the thread num setting (64)
    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    dim3 griddim(nblocks, num_lcs, num_pdots);

    // determine shared memory allocation size
    size_t numBytes = sizeof(fod)*num_pts_max
                      + sizeof(int)*num_pts_max
                      + sizeof(fod)*phase_bins*NUM_THREADS
                      + sizeof(fod)*phase_bins*NUM_THREADS;

    kernel<<<griddim, NUM_THREADS, numBytes>>>(d_ce_vals,
                                               d_freqs, num_freqs,
                                               d_pdots,
                                               num_pdots,
                                               d_mag_bin_inds,
                                               d_time_vals,
                                               d_num_pts_arr,
                                               num_pts_max,
                                               mag_bins,
                                               phase_bins,
                                               num_lcs,
                                               half_dbins);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

}


__global__ void kernel_share(fod* __restrict__ ce_vals, fod* __restrict__ freqs, int num_freqs, fod* __restrict__ pdots,
                       int num_pdots, int* __restrict__ mag_bin_inds,
                       fod* __restrict__ time_vals, int * __restrict__ num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs, fod half_dbins){

       // prepare dynamically allocated shared memory
       extern __shared__ fod overall_phase_prob[];
       fod *bin_counts = &overall_phase_prob[phase_bins];

       // shared quantities for atomicAdd
       __shared__ fod total_points;
       __shared__ fod sum_ij;

       __shared__ fod period, freq, pdot;

       __syncthreads();


       for (int lc_i = blockIdx.y;
            lc_i < num_lcs;
            lc_i += gridDim.y) {

        int num_pts_this_lc = num_pts_arr[lc_i];

        //printf("%d\n", num_pts_this_lc);

      for (int pdot_i = blockIdx.z;
           pdot_i < num_pdots;
           pdot_i += gridDim.z) {


       for (int i = blockIdx.x;
            i < num_freqs;
            i += gridDim.x) {

        // initialize all quantities

        if (threadIdx.x == 0){
        pdot = pdots[pdot_i];
        freq = freqs[i];
        period = 1/freq;
        total_points = 0.0;
        sum_ij = 0.0;

    }
        __syncthreads();
        for (int jj=threadIdx.x; jj<phase_bins*mag_bins; jj+=blockDim.x){
            bin_counts[jj] = 0.0;
        }

        __syncthreads();

        for (int jj=threadIdx.x; jj<phase_bins; jj+=blockDim.x){
            overall_phase_prob[jj] = 0.0;
        }

        __syncthreads();

        for (int jj=threadIdx.x; jj<num_pts_this_lc; jj+=blockDim.x){

            // get phase bin
            int j = get_phase_bin(time_vals[lc_i*num_pts_max + jj], pdot, freq, period, half_dbins, phase_bins);
            int mag_ind = mag_bin_inds[lc_i*num_pts_max + jj];

            // fill histogram
            atomicAdd(&overall_phase_prob[j], 1.0);
            atomicAdd(&bin_counts[j*mag_bins + mag_ind], 1.0);
            atomicAdd(&total_points, 1.0);

            int j2 = (j <= 0) ? j = phase_bins - 1 : (j - 1) % phase_bins ;
            atomicAdd(&overall_phase_prob[j2], 1.0);
            atomicAdd(&bin_counts[j2*mag_bins + mag_ind], 1.0);
            atomicAdd(&total_points, 1.0);

            // check if there should be magnitude overlap
            if (mag_ind != 0 && mag_ind != mag_bins - 1){

                mag_ind -= 1;

                atomicAdd(&overall_phase_prob[j], 1.0);
                atomicAdd(&bin_counts[j*mag_bins + mag_ind], 1.0);
                atomicAdd(&total_points, 1.0);

                atomicAdd(&overall_phase_prob[j2], 1.0);
                atomicAdd(&bin_counts[j2*mag_bins + mag_ind], 1.0);
                atomicAdd(&total_points, 1.0);
            }
        }

        __syncthreads();

        // SUM all bins in the 2D histogram
        for (int jj=threadIdx.x; jj<phase_bins*mag_bins; jj+=blockDim.x){
            // this will create nans if bin_counts[jj] is 0.0
            if (bin_counts[jj] == 0.0) continue;
            int j = jj/mag_bins;

            atomicAdd(&sum_ij, bin_counts[jj]*log(overall_phase_prob[j]/bin_counts[jj]));
        }
        __syncthreads();
        ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = sum_ij/total_points;

     }
   }
   }
   }


// Wrapper function for the long gce version
//callable from another script or cython
   void run_long_lc_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
                int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins)
{

    // Set grid: each freq and pdot are given singular block,
    // rather than thread
    dim3 griddim(num_freqs, num_lcs, num_pdots);

    // number o
    int NUM_THREADS_LONG = 256;

    // get size of shared memory
    size_t numBytes = phase_bins*sizeof(fod) + mag_bins*phase_bins*sizeof(fod);

    kernel_share<<<griddim, NUM_THREADS_LONG, numBytes>>>(d_ce_vals, d_freqs, num_freqs, d_pdots, num_pdots,
                                 d_mag_bin_inds, d_time_vals,
                                 d_num_pts_arr, num_pts_max, mag_bins, phase_bins,
                                 num_lcs, half_dbins);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

}
