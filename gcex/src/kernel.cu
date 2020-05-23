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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__
double mod_func(double x, double y)
{
    int num = (int) (x/y);

    double check = x - (num*y);

    return check;
}


__device__
int get_phase_bin(fod t_val, fod pdot, fod frequency, fod period, fod half_dbins, int phase_bins){
    double folded_val = mod_func((double) t_val-0.5*pdot*frequency*(t_val*t_val), (double)period)*((double)frequency); // between 0 and 1
    if (folded_val < 0.0) folded_val *= -1.0;

    int j = (int) abs((folded_val/(double)half_dbins));
    //printf("1: %d %d %lf %lf\n", j, phase_bins, folded_val, half_dbins);
    j = j % phase_bins;
    //else if (j>phase_bins)  j = phase_bins;
    //printf("2: %d %d %lf %lf\n", j, phase_bins, folded_val, half_dbins);
    return j;
}

__device__ fod ce (fod frequency, fod pdot, fod* __restrict__ phase_bin_edges,
                   int* __restrict__ mag_bin_inds, fod* __restrict__ time_vals, int npoints,
                   int mag_bins, int phase_bins,
                   int offset, int lc_i, fod lc_start_time,
                   fod * temp_phase_prob, fod *overall_phase_prob, fod half_dbins){
    fod period = 1./frequency;
    fod folded_val = 0.0;
    int mag_ind_1 = -1, mag_ind_2=-1;
    int l = 1;
    fod sum_ij = 0.0;
    fod t_val = 0.0;
    fod total_points = 0.0;
    int j = 0;

    for (int jj = 0; jj<phase_bins; jj+=1){
            overall_phase_prob[jj] = 0.0;
            temp_phase_prob[jj] = 0.0;
        }

    //for (int i=0; i<mag_bins; i++){
    //    mag_bin_vals[i] = 0;
    //}

    //fod current_phase_prob = 0.0;
    int temp_mag_points = 0;
    short index=0;
    int ind_mag=0;

    int temp_ind = 0;
    int gap = 0;
    for (int k=0; k<npoints; k++){

        /*# if __CUDA_ARCH__>=200
        if ((offset == 0)  && (lc_i == 0)){
            printf("(%d %d %d): %lf - %lf\n", lc_i, j, offset, phase_bin_edges[2*j], phase_bin_edges[2*j+1]);
        }
        #endif //*/
        //int k = 0;
        //while(k < npoints - gap){

            ///index = indicies[k];
            //if (index == -1) break;

            t_val = time_vals[k] - lc_start_time;
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            overall_phase_prob[j] += 1.0;
            total_points += 1.0;

            //if (k == 10) printf("%lf %lf %lf %lf, %d %d\n", t_val, pdot, period, half_dbins, j, phase_bins);

            j = (j-1) % phase_bins;
            overall_phase_prob[j] += 1.0;
            total_points += 1.0;
        }

    ind_mag = mag_bin_inds[0];
    int kk = 0;
    while (kk < npoints){
        while ((mag_bin_inds[kk] == ind_mag)){
            t_val = time_vals[kk] - lc_start_time;
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            temp_phase_prob[j] += 1.0;
            //printf("%lf %lf %lf %lf, %d %d %d\n", t_val, pdot, period, half_dbins, j, mag_bin_inds[kk], ind_mag);
            j = (j-1) % phase_bins;
            temp_phase_prob[j] += 1.0;

            kk += 1;
            if (kk == npoints) break;
        }

        for (int jj = 0; jj<phase_bins; jj+=1){
            if ((overall_phase_prob[jj] > 0.0) && (temp_phase_prob[jj] > 0.0)){
                 //printf("%d %d %lf %lf \n", ind_mag, jj, overall_phase_prob[jj], temp_phase_prob[jj]);
                sum_ij += temp_phase_prob[jj]*log(overall_phase_prob[jj]/temp_phase_prob[jj]);
                temp_phase_prob[jj] = 0.0;
            }
        }
        if (kk < npoints) ind_mag = mag_bin_inds[kk];
    }
                //printf("%d, %lf, %lf\n", l, mag_bins[l-1], current_phase_prob);

            //        indicies[k] = indicies[(npoints - 1) - gap];
            //        gap += 1;
            //}
            //else k++;
        //}
        /*# if __CUDA_ARCH__>=200
        if ((offset == mag_bins*255)  && (lc_i == 0)){
            printf("(%d %d %d %.18e %e)\n", lc_i, j, offset, frequency, current_phase_prob);
            for (int jj=0; jj<mag_bins; jj++) printf("%e, ", mag_bin_vals[offset + jj]);
            printf("\n");
            for (int jj=0; jj<mag_bins+1; jj++) printf("%e, ", mag_bin_edges[jj]);
            printf("\n\n");
        }
        #endif //*/


    //for (int j=0; j<phase_bins*mag_bins; j++){
        //if (current_phase_prob[j] == 0.0) continue;
        //fod curr_phase_p = current_phase_prob[j];
        //for (int i=0; i<mag_bins; i++){
            //if (bin_counter[j*mag_bins + i] > 0.0){
        //        continue;
                //if (j < 2) printf("%d, %d, %lf, %lf\n", j, i,bin_counter[j*mag_bins + i], curr_phase_p);
            //    sum_ij += (bin_counter[j*mag_bins + i])*log((curr_phase_p)/(bin_counter[j*mag_bins + i]));
        //    }

        //}
    /*# if __CUDA_ARCH__>=200
    if ((offset == mag_bins*255)  && (lc_i == 0)){
        printf("(%d %d %.18e CE: %.18e)\n", lc_i, offset, frequency, sum_ij);
    }
    #endif //*/
    //if (lc_i == 1) printf("%lf %lf %d\n", sum_ij, total_points, npoints);
    return  sum_ij/((fod) total_points);
    //return 0.0;
}

__global__ void kernel(fod* __restrict__ ce_vals, fod* __restrict__ freqs, int num_freqs, fod* __restrict__ pdots,
                       int num_pdots, fod* __restrict__ phase_bin_edges, int* __restrict__ mag_bin_inds,
                       fod* __restrict__ time_vals, int * __restrict__ num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs,
                       fod* __restrict__ min_light_curve_times, fod half_dbins){


    // __shared__ fod share_mag_bin_vals[NUM_THREADS*10];
    extern __shared__ fod share_bins_phase[];
    __shared__ fod time_vals_share[2000];
    __shared__ int mag_bin_inds_share[2000];
    __shared__ fod temp_phase_prob[51*NUM_THREADS];
    __shared__ fod overall_phase_prob[51*NUM_THREADS];


    //for (int j=threadIdx.x; j<=phase_bins; j+=blockDim.x){
    //    share_bins_phase[2*j] = phase_bin_edges[2*j];
    //    share_bins_phase[2*j+1] = phase_bin_edges[2*j+1];
  // }

    __syncthreads();


    for (int lc_i = blockIdx.y;
         lc_i < num_lcs;
         lc_i += gridDim.y) {

     int num_pts_this_lc = num_pts_arr[lc_i];
     fod lc_start_time = min_light_curve_times[lc_i];

     //printf("%d\n", num_pts_this_lc);

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

    // TODO: make this adjustable: https://devblogs.nvidia.com/using-shared-memory-cuda-cc/  !!!!
    int offset = mag_bins*threadIdx.x;


    //__shared__ fod share_t_vals[100];
    //__shared__ fod share_magnitude[100];

        //for (int j=0; j<num_pts_this_lc; j++) share_t_vals[j] = time_vals[lc_i*num_pts_max + j];
        //for (int j=0; j<num_pts_this_lc; j++) share_magnitude[j] = mag_vals[lc_i*num_pts_max + j];
        /*# if __CUDA_ARCH__>=200
        if ((freqs[i] < 2.314814814814814839e-07) && (lc_i == 1)){
            printf("%.18e\n", freqs[i]);
            for (int jj=0; jj<=mag_bins; jj++) printf("%e, ", share_bins_mag[jj]);
            printf("\n\n");
        }


        #endif //*/

    ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = ce(freqs[i], pdots[pdot_i], share_bins_phase, &mag_bin_inds[0],
                                                          &time_vals_share[0], num_pts_this_lc, mag_bins, phase_bins,
                                                          offset, lc_i, lc_start_time,
                                                          &temp_phase_prob[threadIdx.x*51], &overall_phase_prob[threadIdx.x*51], half_dbins);
  }
}
}
}

void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, fod *d_phase_bin_edges, int *d_mag_bin_inds, fod *d_time_vals,
             int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod *d_min_light_curve_times, fod half_dbins)
{
    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    //printf("%d\n", nblocks, num_pdots, num_lcs);
    dim3 griddim(nblocks, num_lcs, num_pdots);

    //cudaStream_t streams[num_lcs];

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
        //cudaStreamCreate(&streams[lc_i]);

        kernel<<<griddim, NUM_THREADS>>>(d_ce_vals, d_freqs, num_freqs, d_pdots, num_pdots,
                                     d_phase_bin_edges, d_mag_bin_inds, d_time_vals,
                                     d_num_pts_arr, num_pts_max, mag_bins, phase_bins,
                                     num_lcs, d_min_light_curve_times, half_dbins);
    //}
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
    //        cudaStreamDestroy(streams[lc_i]);

    //    }

}
