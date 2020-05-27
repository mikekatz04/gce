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

__device__ fod ce (fod frequency, fod pdot,
                   int* __restrict__ mag_bin_inds, fod* __restrict__ time_vals, int npoints,
                   int mag_bins, int phase_bins,
                   int offset, int lc_i,
                   fod * temp_phase_prob, fod *overall_phase_prob, fod half_dbins){
    fod period = 1./frequency;
    fod folded_val = 0.0;
    int mag_ind_1 = -1, mag_ind_2=-1;
    int l = 1;
    fod sum_ij = 0.0;
    fod t_val = 0.0;
    fod total_points = 0.0;
    int j = 0;
    int j1 = 0;

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

            t_val = time_vals[k];
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            for (int i=0; i<2; i+=1){

            }

            overall_phase_prob[j] += 1.0;
            total_points += 1.0;

            //if (k == 10) printf("%lf %lf %lf %lf, %d %d\n", t_val, pdot, period, half_dbins, j, phase_bins);

            int j1 = (j > 0) ? j = phase_bins - 1 : (j - 1) % phase_bins;
            overall_phase_prob[j1] += 1.0;
            total_points += 1.0;

            ind_mag = mag_bin_inds[k];
            if ((ind_mag != 0) && (ind_mag != mag_bins - 1)){

                overall_phase_prob[j] += 1.0;
                total_points += 1.0;

                overall_phase_prob[j1] += 1.0;
                total_points += 1.0;

            }

        }

    ind_mag = mag_bin_inds[0];
    int kk = 0;
    while (kk < npoints){
        while ((mag_bin_inds[kk] == ind_mag)){
            t_val = time_vals[kk];
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            temp_phase_prob[j] += 1.0;
            //printf("%lf %lf %lf %lf, %d %d %d\n", t_val, pdot, period, half_dbins, j, mag_bin_inds[kk], ind_mag);
            int j1 = (j <= 0) ? j = phase_bins - 1 : (j - 1) % phase_bins ;
            temp_phase_prob[j1] += 1.0;

            if ((ind_mag != 0) && (ind_mag != mag_bins - 1)){

                overall_phase_prob[j] += 1.0;
                total_points += 1.0;

                overall_phase_prob[j1] += 1.0;
                total_points += 1.0;

            }

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
                       int num_pdots, int* __restrict__ mag_bin_inds,
                       fod* __restrict__ time_vals, int * __restrict__ num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs, fod half_dbins){


    // __shared__ fod share_mag_bin_vals[NUM_THREADS*10];
    extern __shared__ fod time_vals_share[];
    int *mag_bin_inds_share = (int*)(&time_vals_share[num_pts_max*2]);
    //printf("%ld %ld, %ld\n", time_vals_share, mag_bin_inds_share, num_pts_max*2*sizeof(fod));
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

     //printf("%d\n", num_pts_this_lc);

     for (int j=threadIdx.x; j<num_pts_this_lc; j+=blockDim.x){

         time_vals_share[j] = time_vals[lc_i*num_pts_max + j];
         mag_bin_inds_share[j] = mag_bin_inds[lc_i*num_pts_max + j];

         //printf("%d\n", mag_bin_inds_share[j]);
     }

      //printf("CHECK\n");

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

    ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = ce(freqs[i], pdots[pdot_i], mag_bin_inds_share,
                                                          &time_vals_share[0], num_pts_this_lc, mag_bins, phase_bins,
                                                          offset, lc_i,
                                                          &temp_phase_prob[threadIdx.x*51], &overall_phase_prob[threadIdx.x*51], half_dbins);
  }
}
}
}

void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
             int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins)
{
    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    //printf("%d\n", nblocks, num_pdots, num_lcs);
    dim3 griddim(nblocks, num_lcs, num_pdots);

    //cudaStream_t streams[num_lcs];

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
        //cudaStreamCreate(&streams[lc_i]);

        size_t numBytes = 2*(sizeof(fod)*num_pts_max + sizeof(int)*num_pts_max);
        //printf("num_bytes: %d, %d, %d\n", numBytes, sizeof(int), num_pts_max*2*sizeof(fod));
        kernel<<<griddim, NUM_THREADS, numBytes>>>(d_ce_vals, d_freqs, num_freqs, d_pdots, num_pdots, d_mag_bin_inds, d_time_vals,
                                     d_num_pts_arr, num_pts_max, mag_bins, phase_bins,
                                     num_lcs, half_dbins);
    //}
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
    //        cudaStreamDestroy(streams[lc_i]);

    //    }

}


__global__ void kernel_share(fod* __restrict__ ce_vals, fod* __restrict__ freqs, int num_freqs, fod* __restrict__ pdots,
                       int num_pdots, int* __restrict__ mag_bin_inds,
                       fod* __restrict__ time_vals, int * __restrict__ num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs, fod half_dbins){

       // __shared__ fod share_mag_bin_vals[NUM_THREADS*10];
//       extern __shared__ fod share_bins_phase[];
       __shared__ fod overall_phase_prob[15];
       __shared__ fod bin_counts[10*15];

       __shared__ fod total_points;
       __shared__ fod sum_ij;

       __shared__ fod period, freq, pdot;

       //for (int j=threadIdx.x; j<=phase_bins; j+=blockDim.x){
       //    share_bins_phase[2*j] = phase_bin_edges[2*j];
       //    share_bins_phase[2*j+1] = phase_bin_edges[2*j+1];
     // }


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

        if (threadIdx.x == 0){
        pdot = pdots[pdot_i];
        freq = freqs[i];
        period = 1/freq;
        total_points = 0.0;
        sum_ij = 0.0;

    }
        __syncthreads();
        for (int jj=threadIdx.x; jj<15*10; jj+=blockDim.x){
            bin_counts[jj] = 0.0;
        }

        __syncthreads();

        for (int jj=threadIdx.x; jj<15; jj+=blockDim.x){
            overall_phase_prob[jj] = 0.0;
        }

        __syncthreads();

        for (int jj=threadIdx.x; jj<num_pts_this_lc; jj+=blockDim.x){

            int j = get_phase_bin(time_vals[lc_i*num_pts_max + jj], pdot, freq, period, half_dbins, phase_bins);
            int mag_ind = mag_bin_inds[lc_i*num_pts_max + jj];

            atomicAdd(&overall_phase_prob[j], 1.0);
            atomicAdd(&bin_counts[j*10 + mag_ind], 1.0);
            atomicAdd(&total_points, 1.0);

            int j2 = (j >= 0) ? j = 14 : (j - 1) % 15 ;  // FIXME: when adjusted from 15, needs to be adjusted.
            atomicAdd(&overall_phase_prob[j2], 1.0);
            atomicAdd(&bin_counts[j2*10 + mag_ind], 1.0);
            atomicAdd(&total_points, 1.0);

            if (mag_ind != 0 && mag_ind != mag_bins - 1){

                mag_ind -= 1;

                atomicAdd(&overall_phase_prob[j], 1.0);
                atomicAdd(&bin_counts[j*10 + mag_ind], 1.0);
                atomicAdd(&total_points, 1.0);

                atomicAdd(&overall_phase_prob[j2], 1.0);
                atomicAdd(&bin_counts[j2*10 + mag_ind], 1.0);
                atomicAdd(&total_points, 1.0);
            }

            //if (i == 950) printf("%d %d %d %d %d %d, %lf %lf\n",lc_i, pdot_i, i, jj, j, mag_ind, overall_phase_prob[j], bin_counts[j*10 + mag_ind]);

        }

        __syncthreads();

        //fod sum_ij = 0.0;
        for (int jj=threadIdx.x; jj<15*10; jj+=blockDim.x){
            if (bin_counts[jj] == 0.0) continue;
            int j = jj/10;

            //printf("%d %d, %lf %lf\n", jj, j, overall_phase_prob[j], bin_counts[jj]);
            //if (overall_phase_prob[j] <bin_counts[jj]) printf("bad %d %d %d %d %d, %lf %lf\n",lc_i, pdot_i, i, jj, j, overall_phase_prob[j], bin_counts[jj]);
            atomicAdd(&sum_ij, bin_counts[jj]*log(overall_phase_prob[j]/bin_counts[jj]));
        }
        __syncthreads();
        //if (i == 0) printf("%lf %lf\n", sum_ij, total_points);

        ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = sum_ij/total_points;

     }
   }
   }
   }


   void run_long_lc_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
                int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins)
{
    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    //printf("%d\n", nblocks, num_pdots, num_lcs);
    //dim3 griddim(nblocks, num_lcs, num_pdots);
    dim3 griddim(num_freqs, num_lcs, num_pdots);

    int nthreads_long = 256;
    kernel_share<<<griddim, nthreads_long>>>(d_ce_vals, d_freqs, num_freqs, d_pdots, num_pdots,
                                 d_mag_bin_inds, d_time_vals,
                                 d_num_pts_arr, num_pts_max, mag_bins, phase_bins,
                                 num_lcs, half_dbins);
    //cudaStream_t streams[num_lcs];

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
        //cudaStreamCreate(&streams[lc_i]);
    //}
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    //for (int lc_i=0; lc_i<num_lcs; lc_i+=1){
    //        cudaStreamDestroy(streams[lc_i]);

    //    }

}
