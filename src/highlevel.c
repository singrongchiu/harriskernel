#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "nonmaxsup.h"

#define K 0.04   // Harris detector constant
// #define THRESHOLD 1e4   // Corner response threshold

// gcc -O2 -std=c11 -mavx2 -o highlevel highlevel.c -lm
// NEW: linked with nonmaxsup:
// gcc -O2 -std=c11 -mavx2 -o highlevel highlevel.c nonmaxsup.c -lm
// gcc highlevel.c nonmaxsup.c -O3 -mavx2 -mfma -lm -o highlevel

float** alloc_matrix(int h, int w) {
    float **m = malloc(h * sizeof(float*));
    for (int i = 0; i < h; i++)
        m[i] = calloc(w, sizeof(float));
    return m;
}

void free_matrix(float **m, int h) {
    for (int i = 0; i < h; i++) free(m[i]);
    free(m);
}


// void sobel(float **img, float **Ix, float **Iy, int h, int w) {
//     int gx[3][3] = {
//         {-1, 0, 1},
//         {-2, 0, 2},
//         {-1, 0, 1}
//     };

//     int gy[3][3] = {
//         {-1,-2,-1},
//         { 0, 0, 0},
//         { 1, 2, 1}
//     };

//     for (int y = 1; y < h - 1; y++) {
//         for (int x = 1; x < w - 1; x++) {
//             float sx = 0, sy = 0;
//             for (int j = 0; j < 3; j++)
//                 for (int i = 0; i < 3; i++) {
//                     sx += img[y + j - 1][x + i - 1] * gx[j][i];
//                     sy += img[y + j - 1][x + i - 1] * gy[j][i];
//                 }
//             Ix[y][x] = sx;
//             Iy[y][x] = sy;
//         }
//     }
// }

// void sobel_avx2_fma(float **img, float **Ix, float **Iy, int h, int w)
// {
//     // broadcast coefficients
//     const __m256 k_gx_m1 = _mm256_set1_ps(-1.0f);
//     // const __m256 k_gx_0  = _mm256_set1_ps( 0.0f);
//     const __m256 k_gx_p1 = _mm256_set1_ps( 1.0f);

//     const __m256 k_gx_m2 = _mm256_set1_ps(-2.0f);
//     const __m256 k_gx_p2 = _mm256_set1_ps( 2.0f);

//     // Broadcast Sobel coefficients for gy
//     const __m256 k_gy_m1 = _mm256_set1_ps(-1.0f);
//     const __m256 k_gy_m2 = _mm256_set1_ps(-2.0f);
//     // const __m256 k_gy_0  = _mm256_set1_ps( 0.0f);
//     const __m256 k_gy_p1 = _mm256_set1_ps( 1.0f);
//     const __m256 k_gy_p2 = _mm256_set1_ps( 2.0f);

//     for (int y = 1; y < h - 1; y++)
//     {
//         for (int x = 1; x < w - 8; x += 8)
//         {
//             // Load rows around (y, x)
//             __m256 r0_l = _mm256_loadu_ps(&img[y-1][x-1]);
//             __m256 r0_m = _mm256_loadu_ps(&img[y-1][x  ]);
//             __m256 r0_r = _mm256_loadu_ps(&img[y-1][x+1]);

//             __m256 r1_l = _mm256_loadu_ps(&img[y  ][x-1]);
//             __m256 r1_m = _mm256_loadu_ps(&img[y  ][x  ]);
//             __m256 r1_r = _mm256_loadu_ps(&img[y  ][x+1]);

//             __m256 r2_l = _mm256_loadu_ps(&img[y+1][x-1]);
//             __m256 r2_m = _mm256_loadu_ps(&img[y+1][x  ]);
//             __m256 r2_r = _mm256_loadu_ps(&img[y+1][x+1]);

//             // top row gx
//             __m256 gx = _mm256_fmadd_ps(r0_r, k_gx_p1, _mm256_mul_ps(r0_l, k_gx_m1));
            
//             gx = _mm256_fmadd_ps(r1_r, k_gx_p2, _mm256_fmadd_ps(r1_l, k_gx_m2, gx));
            
//             gx = _mm256_fmadd_ps(r2_r, k_gx_p1,
//                                  _mm256_fmadd_ps(r2_l, k_gx_m1, gx));
            
//             // top row gy
//             __m256 gy = _mm256_fmadd_ps(r0_r, k_gy_m1,
//                         _mm256_fmadd_ps(r0_m, k_gy_m2,
//                                         _mm256_mul_ps(r0_l, k_gy_m1)));
            
//             gy = _mm256_fmadd_ps(r2_r, k_gy_p1, _mm256_fmadd_ps(r2_m, k_gy_p2,
//                                  _mm256_fmadd_ps(r2_l, k_gy_p1, gy)));
            
//             _mm256_storeu_ps(&Ix[y][x], gx);
//             _mm256_storeu_ps(&Iy[y][x], gy);
//         }
//     }
// }

void sobel_avx_simple(float **img, float **Ix, float **Iy, int h, int w)
{
    // Only 4 registers for coefficients
    const __m256 k_m1 = _mm256_set1_ps(-1.0f);
    const __m256 k_p1 = _mm256_set1_ps( 1.0f);
    const __m256 k_m2 = _mm256_set1_ps(-2.0f);
    const __m256 k_p2 = _mm256_set1_ps( 2.0f);

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x <= w - 48; x += 48) { // 6 blocks of 8 pixels
            // initialize accumulators
            __m256 gx0 = _mm256_setzero_ps(), gx1 = _mm256_setzero_ps(),
                   gx2 = _mm256_setzero_ps(), gx3 = _mm256_setzero_ps(),
                   gx4 = _mm256_setzero_ps(), gx5 = _mm256_setzero_ps();

            __m256 gy0 = _mm256_setzero_ps(), gy1 = _mm256_setzero_ps(),
                   gy2 = _mm256_setzero_ps(), gy3 = _mm256_setzero_ps(),
                   gy4 = _mm256_setzero_ps(), gy5 = _mm256_setzero_ps();

            for (int j = -1; j <= 1; j++) {
                float *row = img[y + j];

                __m256 r0_l0 = _mm256_loadu_ps(row + x-1 + 0*8);
                __m256 r0_m0 = _mm256_loadu_ps(row + x   + 0*8);
                __m256 r0_r0 = _mm256_loadu_ps(row + x+1 + 0*8);
                __m256 r0_l1 = _mm256_loadu_ps(row + x-1 + 1*8);
                __m256 r0_m1 = _mm256_loadu_ps(row + x   + 1*8);
                __m256 r0_r1 = _mm256_loadu_ps(row + x+1 + 1*8);
                __m256 r0_l2 = _mm256_loadu_ps(row + x-1 + 2*8);
                __m256 r0_m2 = _mm256_loadu_ps(row + x   + 2*8);
                __m256 r0_r2 = _mm256_loadu_ps(row + x+1 + 2*8);
                __m256 r0_l3 = _mm256_loadu_ps(row + x-1 + 3*8);
                __m256 r0_m3 = _mm256_loadu_ps(row + x   + 3*8);
                __m256 r0_r3 = _mm256_loadu_ps(row + x+1 + 3*8);
                __m256 r0_l4 = _mm256_loadu_ps(row + x-1 + 4*8);
                __m256 r0_m4 = _mm256_loadu_ps(row + x   + 4*8);
                __m256 r0_r4 = _mm256_loadu_ps(row + x+1 + 4*8);
                __m256 r0_l5 = _mm256_loadu_ps(row + x-1 + 5*8);
                __m256 r0_m5 = _mm256_loadu_ps(row + x   + 5*8);
                __m256 r0_r5 = _mm256_loadu_ps(row + x+1 + 5*8);

                // gx
                if (j == -1) {
                    gx0 = _mm256_fmadd_ps(r0_l0, k_m1, _mm256_fmadd_ps(r0_r0, k_p1, gx0));
                    gx1 = _mm256_fmadd_ps(r0_l1, k_m1, _mm256_fmadd_ps(r0_r1, k_p1, gx1));
                    gx2 = _mm256_fmadd_ps(r0_l2, k_m1, _mm256_fmadd_ps(r0_r2, k_p1, gx2));
                    gx3 = _mm256_fmadd_ps(r0_l3, k_m1, _mm256_fmadd_ps(r0_r3, k_p1, gx3));
                    gx4 = _mm256_fmadd_ps(r0_l4, k_m1, _mm256_fmadd_ps(r0_r4, k_p1, gx4));
                    gx5 = _mm256_fmadd_ps(r0_l5, k_m1, _mm256_fmadd_ps(r0_r5, k_p1, gx5));

                    gy0 = _mm256_fmadd_ps(r0_l0, k_m1, _mm256_fmadd_ps(r0_m0, k_m2, _mm256_fmadd_ps(r0_r0, k_m1, gy0)));
                    gy1 = _mm256_fmadd_ps(r0_l1, k_m1, _mm256_fmadd_ps(r0_m1, k_m2, _mm256_fmadd_ps(r0_r1, k_m1, gy1)));
                    gy2 = _mm256_fmadd_ps(r0_l2, k_m1, _mm256_fmadd_ps(r0_m2, k_m2, _mm256_fmadd_ps(r0_r2, k_m1, gy2)));
                    gy3 = _mm256_fmadd_ps(r0_l3, k_m1, _mm256_fmadd_ps(r0_m3, k_m2, _mm256_fmadd_ps(r0_r3, k_m1, gy3)));
                    gy4 = _mm256_fmadd_ps(r0_l4, k_m1, _mm256_fmadd_ps(r0_m4, k_m2, _mm256_fmadd_ps(r0_r4, k_m1, gy4)));
                    gy5 = _mm256_fmadd_ps(r0_l5, k_m1, _mm256_fmadd_ps(r0_m5, k_m2, _mm256_fmadd_ps(r0_r5, k_m1, gy5)));
                }
                else if (j == 0) {
                    gx0 = _mm256_fmadd_ps(r0_l0, k_m2, _mm256_fmadd_ps(r0_r0, k_p2, gx0));
                    gx1 = _mm256_fmadd_ps(r0_l1, k_m2, _mm256_fmadd_ps(r0_r1, k_p2, gx1));
                    gx2 = _mm256_fmadd_ps(r0_l2, k_m2, _mm256_fmadd_ps(r0_r2, k_p2, gx2));
                    gx3 = _mm256_fmadd_ps(r0_l3, k_m2, _mm256_fmadd_ps(r0_r3, k_p2, gx3));
                    gx4 = _mm256_fmadd_ps(r0_l4, k_m2, _mm256_fmadd_ps(r0_r4, k_p2, gx4));
                    gx5 = _mm256_fmadd_ps(r0_l5, k_m2, _mm256_fmadd_ps(r0_r5, k_p2, gx5));
                    // gy middle row is 0
                }
                else { // j == 1
                    gx0 = _mm256_fmadd_ps(r0_l0, k_m1, _mm256_fmadd_ps(r0_r0, k_p1, gx0));
                    gx1 = _mm256_fmadd_ps(r0_l1, k_m1, _mm256_fmadd_ps(r0_r1, k_p1, gx1));
                    gx2 = _mm256_fmadd_ps(r0_l2, k_m1, _mm256_fmadd_ps(r0_r2, k_p1, gx2));
                    gx3 = _mm256_fmadd_ps(r0_l3, k_m1, _mm256_fmadd_ps(r0_r3, k_p1, gx3));
                    gx4 = _mm256_fmadd_ps(r0_l4, k_m1, _mm256_fmadd_ps(r0_r4, k_p1, gx4));
                    gx5 = _mm256_fmadd_ps(r0_l5, k_m1, _mm256_fmadd_ps(r0_r5, k_p1, gx5));

                    gy0 = _mm256_fmadd_ps(r0_l0, k_p1, _mm256_fmadd_ps(r0_m0, k_p2, _mm256_fmadd_ps(r0_r0, k_p1, gy0)));
                    gy1 = _mm256_fmadd_ps(r0_l1, k_p1, _mm256_fmadd_ps(r0_m1, k_p2, _mm256_fmadd_ps(r0_r1, k_p1, gy1)));
                    gy2 = _mm256_fmadd_ps(r0_l2, k_p1, _mm256_fmadd_ps(r0_m2, k_p2, _mm256_fmadd_ps(r0_r2, k_p1, gy2)));
                    gy3 = _mm256_fmadd_ps(r0_l3, k_p1, _mm256_fmadd_ps(r0_m3, k_p2, _mm256_fmadd_ps(r0_r3, k_p1, gy3)));
                    gy4 = _mm256_fmadd_ps(r0_l4, k_p1, _mm256_fmadd_ps(r0_m4, k_p2, _mm256_fmadd_ps(r0_r4, k_p1, gy4)));
                    gy5 = _mm256_fmadd_ps(r0_l5, k_p1, _mm256_fmadd_ps(r0_m5, k_p2, _mm256_fmadd_ps(r0_r5, k_p1, gy5)));
                }
            }

            // Store results
            _mm256_storeu_ps(&Ix[y][x + 0*8], gx0);
            _mm256_storeu_ps(&Ix[y][x + 1*8], gx1);
            _mm256_storeu_ps(&Ix[y][x + 2*8], gx2);
            _mm256_storeu_ps(&Ix[y][x + 3*8], gx3);
            _mm256_storeu_ps(&Ix[y][x + 4*8], gx4);
            _mm256_storeu_ps(&Ix[y][x + 5*8], gx5);

            _mm256_storeu_ps(&Iy[y][x + 0*8], gy0);
            _mm256_storeu_ps(&Iy[y][x + 1*8], gy1);
            _mm256_storeu_ps(&Iy[y][x + 2*8], gy2);
            _mm256_storeu_ps(&Iy[y][x + 3*8], gy3);
            _mm256_storeu_ps(&Iy[y][x + 4*8], gy4);
            _mm256_storeu_ps(&Iy[y][x + 5*8], gy5);
        }
    }
}


// void gaussian3(float **src, float **dst, int h, int w) {
//     float g[3][3] = {
//         {1, 2, 1},
//         {2, 4, 2},
//         {1, 2, 1}
//     };
//     float norm = 16.0;

//     for (int y = 1; y < h - 1; y++) {
//         for (int x = 1; x < w - 1; x++) {
//             float sum = 0;
//             for (int j = 0; j < 3; j++)
//                 for (int i = 0; i < 3; i++)
//                     sum += src[y + j - 1][x + i - 1] * g[j][i];

//             dst[y][x] = sum / norm;
//         }
//     }
// }

// void gaussian3(float **src, float **dst, int h, int w) {
//     // 3x3 Gaussian kernel with sigma=1, normalized
//     const float k[3][3] = {
//         {1.f/16, 2.f/16, 1.f/16},
//         {2.f/16, 4.f/16, 2.f/16},
//         {1.f/16, 2.f/16, 1.f/16}
//     };

//     // Broadcast kernel values to AVX registers
//     __m256 k00 = _mm256_set1_ps(k[0][0]);
//     __m256 k01 = _mm256_set1_ps(k[0][1]);
//     __m256 k02 = _mm256_set1_ps(k[0][2]);
//     __m256 k10 = _mm256_set1_ps(k[1][0]);
//     __m256 k11 = _mm256_set1_ps(k[1][1]);
//     __m256 k12 = _mm256_set1_ps(k[1][2]);
//     __m256 k20 = _mm256_set1_ps(k[2][0]);
//     __m256 k21 = _mm256_set1_ps(k[2][1]);
//     __m256 k22 = _mm256_set1_ps(k[2][2]);

//     for (int y = 1; y < h - 1; y++) {
//         for (int x = 1; x < w - 8; x += 8) { // process 8 pixels at a time
//             __m256 r0_l = _mm256_loadu_ps(&src[y-1][x-1]);
//             __m256 r0_m = _mm256_loadu_ps(&src[y-1][x]);
//             __m256 r0_r = _mm256_loadu_ps(&src[y-1][x+1]);

//             __m256 r1_l = _mm256_loadu_ps(&src[y][x-1]);
//             __m256 r1_m = _mm256_loadu_ps(&src[y][x]);
//             __m256 r1_r = _mm256_loadu_ps(&src[y][x+1]);

//             __m256 r2_l = _mm256_loadu_ps(&src[y+1][x-1]);
//             __m256 r2_m = _mm256_loadu_ps(&src[y+1][x]);
//             __m256 r2_r = _mm256_loadu_ps(&src[y+1][x+1]);

//             // Apply Gaussian kernel using fused multiply-add
//             __m256 sum = _mm256_setzero_ps();

//             sum = _mm256_fmadd_ps(r0_l, k00, sum);
//             sum = _mm256_fmadd_ps(r0_m, k01, sum);
//             sum = _mm256_fmadd_ps(r0_r, k02, sum);

//             sum = _mm256_fmadd_ps(r1_l, k10, sum);
//             sum = _mm256_fmadd_ps(r1_m, k11, sum);
//             sum = _mm256_fmadd_ps(r1_r, k12, sum);

//             sum = _mm256_fmadd_ps(r2_l, k20, sum);
//             sum = _mm256_fmadd_ps(r2_m, k21, sum);
//             sum = _mm256_fmadd_ps(r2_r, k22, sum);

//             _mm256_storeu_ps(&dst[y][x], sum);
//         }
//     }
// }

void gaussian3(float **src, float **dst, int h, int w) {
    const float k[3][3] = {
        {1.f/16, 2.f/16, 1.f/16},
        {2.f/16, 4.f/16, 2.f/16},
        {1.f/16, 2.f/16, 1.f/16}
    };

    __m256 k00 = _mm256_set1_ps(k[0][0]);
    __m256 k01 = _mm256_set1_ps(k[0][1]);
    __m256 k02 = _mm256_set1_ps(k[0][2]);
    __m256 k10 = _mm256_set1_ps(k[1][0]);
    __m256 k11 = _mm256_set1_ps(k[1][1]);
    __m256 k12 = _mm256_set1_ps(k[1][2]);
    __m256 k20 = _mm256_set1_ps(k[2][0]);
    __m256 k21 = _mm256_set1_ps(k[2][1]);
    __m256 k22 = _mm256_set1_ps(k[2][2]);

    for (int y = 1; y < h-1; y++) {
        int x;
        for (x = 1; x <= w - 48; x += 48) {  // 6 groups of 8 pixels
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            __m256 sum4 = _mm256_setzero_ps();
            __m256 sum5 = _mm256_setzero_ps();

            for (int j = 0; j < 3; j++) {
                float *row = src[y + j - 1];

                // Load left, middle, right pixels for each of the 6 blocks
                __m256 r_l0 = _mm256_loadu_ps(row + x - 1 + 0*8);
                __m256 r_m0 = _mm256_loadu_ps(row + x + 0*8);
                __m256 r_r0 = _mm256_loadu_ps(row + x + 1 + 0*8);

                __m256 r_l1 = _mm256_loadu_ps(row + x - 1 + 1*8);
                __m256 r_m1 = _mm256_loadu_ps(row + x + 1*8);
                __m256 r_r1 = _mm256_loadu_ps(row + x + 1 + 1*8);

                __m256 r_l2 = _mm256_loadu_ps(row + x - 1 + 2*8);
                __m256 r_m2 = _mm256_loadu_ps(row + x + 2*8);
                __m256 r_r2 = _mm256_loadu_ps(row + x + 1 + 2*8);

                __m256 r_l3 = _mm256_loadu_ps(row + x - 1 + 3*8);
                __m256 r_m3 = _mm256_loadu_ps(row + x + 3*8);
                __m256 r_r3 = _mm256_loadu_ps(row + x + 1 + 3*8);

                __m256 r_l4 = _mm256_loadu_ps(row + x - 1 + 4*8);
                __m256 r_m4 = _mm256_loadu_ps(row + x + 4*8);
                __m256 r_r4 = _mm256_loadu_ps(row + x + 1 + 4*8);

                __m256 r_l5 = _mm256_loadu_ps(row + x - 1 + 5*8);
                __m256 r_m5 = _mm256_loadu_ps(row + x + 5*8);
                __m256 r_r5 = _mm256_loadu_ps(row + x + 1 + 5*8);

                // Select kernel for this row
                __m256 k_l, k_m, k_r;
                if (j == 0) { k_l = k00; k_m = k01; k_r = k02; }
                else if (j == 1) { k_l = k10; k_m = k11; k_r = k12; }
                else { k_l = k20; k_m = k21; k_r = k22; }

                // Interleaved multiply-add
                sum0 = _mm256_fmadd_ps(r_l0, k_l, _mm256_fmadd_ps(r_m0, k_m, _mm256_fmadd_ps(r_r0, k_r, sum0)));
                sum1 = _mm256_fmadd_ps(r_l1, k_l, _mm256_fmadd_ps(r_m1, k_m, _mm256_fmadd_ps(r_r1, k_r, sum1)));
                sum2 = _mm256_fmadd_ps(r_l2, k_l, _mm256_fmadd_ps(r_m2, k_m, _mm256_fmadd_ps(r_r2, k_r, sum2)));
                sum3 = _mm256_fmadd_ps(r_l3, k_l, _mm256_fmadd_ps(r_m3, k_m, _mm256_fmadd_ps(r_r3, k_r, sum3)));
                sum4 = _mm256_fmadd_ps(r_l4, k_l, _mm256_fmadd_ps(r_m4, k_m, _mm256_fmadd_ps(r_r4, k_r, sum4)));
                sum5 = _mm256_fmadd_ps(r_l5, k_l, _mm256_fmadd_ps(r_m5, k_m, _mm256_fmadd_ps(r_r5, k_r, sum5)));
            }

            // Store results
            _mm256_storeu_ps(dst[y] + x + 0*8, sum0);
            _mm256_storeu_ps(dst[y] + x + 1*8, sum1);
            _mm256_storeu_ps(dst[y] + x + 2*8, sum2);
            _mm256_storeu_ps(dst[y] + x + 3*8, sum3);
            _mm256_storeu_ps(dst[y] + x + 4*8, sum4);
            _mm256_storeu_ps(dst[y] + x + 5*8, sum5);
        }

        // TODO: handle remaining pixels at row end
    }
}

void harris_response(float **Ix, float **Iy, float **R, int h, int w) {
    float **Ix2 = alloc_matrix(h, w);
    float **Iy2 = alloc_matrix(h, w);
    float **Ixy = alloc_matrix(h, w);

    // Compute products
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            Ix2[y][x] = Ix[y][x] * Ix[y][x];
            Iy2[y][x] = Iy[y][x] * Iy[y][x];
            Ixy[y][x] = Ix[y][x] * Iy[y][x];
        }

    // Smooth
    gaussian3(Ix2, Ix2, h, w);
    gaussian3(Iy2, Iy2, h, w);
    gaussian3(Ixy, Ixy, h, w);

    // Harris response
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float a = Ix2[y][x];
            float b = Ixy[y][x];
            float c = Iy2[y][x];

            float det = a * c - b * b;
            float trace = a + c;
            R[y][x] = det - K * trace * trace;
        }
    }

    free_matrix(Ix2, h);
    free_matrix(Iy2, h);
    free_matrix(Ixy, h);
}

void nonmax(float **R, unsigned char **corners, int h, int w) {
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float v = R[y][x];
            if (v < THRESHOLD) continue;

            int is_max = 1;
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                    if (R[y + j][x + i] > v)
                        is_max = 0;

            corners[y][x] = is_max ? 255 : 0;
        }
    }
}


void harris_corner_detector(float **image, unsigned char **out,
                            int h, int w)
{
    float **Ix = alloc_matrix(h, w);
    float **Iy = alloc_matrix(h, w);
    float **R  = alloc_matrix(h, w);

    sobel_avx_simple(image, Ix, Iy, h, w);
    harris_response(Ix, Iy, R, h, w);
    process_array_avx2(R, out, h, w);

    free_matrix(Ix, h);
    free_matrix(Iy, h);
    free_matrix(R, h);
}

// luminance formula
unsigned char rgb_to_gray(unsigned char r, unsigned char g, unsigned char b) {
    return (unsigned char)(0.299*r + 0.587*g + 0.114*b);
}

float **load_jpg_as_grayscale_f32(const char *filename, int *h, int *w) {
    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    if (!data) {
        printf("Error loading %s\n", filename);
        return NULL;
    }

    *w = width;
    *h = height;

    float **img = malloc((height + 2) * sizeof(float*));
    for (int y = 0; y < height + 2; y++)
        img[y] = malloc((width + 48) * sizeof(float));

    // convert to grayscale
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            unsigned char R = data[idx];
            unsigned char G = data[idx + 1];
            unsigned char B = data[idx + 2];
            img[y][x] = rgb_to_gray(R, G, B);
        }
        for (int x = 0; x < 44; x++)
            img[y][width + x] = 0.0f; // my custom padding
    }
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < width + 44; x++)
            img[height + y][x] = 0.0f; // padding rows at bottom
    }

    stbi_image_free(data);
    return img;
}

int main() {
    int w, h;
    float **gray_img = load_jpg_as_grayscale_f32("chessboard.jpg", &h, &w);

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    unsigned char **corners = malloc(h * sizeof(unsigned char*));
    for (int i = 0; i < h; i++)
        corners[i] = calloc(w, 1);

    st = rdtsc();
    harris_corner_detector(gray_img, corners, h, w);
    et = rdtsc();
    sum += (et-st);

    printf("RDTSC Base Cycles Taken for HARRIS CORNER: %llu\n\r",sum);
    
    unsigned char *outbuf = malloc(h * w);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            outbuf[y * w + x] = corners[y][x];

    // Write to JPEG (quality 95)
    stbi_write_jpg("corners_1e4.jpg", w, h, 1, outbuf, 95);

    printf("Done. Saved corners.jpg\n");

    // Free memory
    for (int i = 0; i < h; i++) {
        free(gray_img[i]);
        free(corners[i]);
    }
    free(gray_img);
    free(corners);
    free(outbuf);

    printf("Done.\n");
}

