#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

void call_kernel(
    int m, int n,
    int m_out, int n_out,
    int ksize,
    int blocksize,          // ignored in SIMD but kept for API
    float *restrict k,
    float *restrict a,
    float *restrict op);

void naive_1d_conv(
    int m,
    float k0,
    float k1,
    float k2,
    float *restrict a,
    float *restrict op_check)
{
    for (int p = 1; p != (m - 1); ++p)
    {
        op_check[p] = a[p - 1] * k0 + a[p] * k1 + a[p + 1] * k2;
        // printf("%d \t %.2f %.2f %.2f %.2f \n", p, op[p], a[p-1], a[p], a[p+1]);
    }
}

void naive_conv(
    int m,
    int n,
    int m_out,
    int n_out,
    int ksize,
    float k0,
    float k1,
    float k2,
    float k3,
    float *restrict a,
    float *restrict op_check
) {
    float kernelvec[4] = {k0, k1, k2, k3};
    for (int i = 0; i != m_out; ++i) {
        for (int j = 0; j != n_out; ++j) {
            for (int k = 0; k != ksize; ++k) {
                op_check[(i)*n_out + j] += kernelvec[k] * a[(i+k)*n + j];
                
            }
            printf("%.2f ", op_check[(i)*n_out + j]);
        }
        printf("\n");
    }
}


// void pre_simd_conv(
//     int m,
//     int n,
//     int m_out,
//     int n_out,
//     int ksize,
//     int blocksize,
//     float k0,
//     float k1,
//     float k2,
//     float k3,
//     float *restrict a,
//     float *restrict op
// ) {
//     float kernelvec[4] = {k0, k1, k2, k3};
//     for (int k=0; k != ksize; ++k) {
//         for (int i=k; i <= (int) (n/blocksize); ++i) {
//             for (int j = 0; j != m_out; ++j) {
//                 for (int l = 0; l != n_out; ++l) {

//                 }
//             }
//         }
//     }

// }


int main()
{
    float *a;
    float *op, *op_check;
    unsigned long long t0, t1;

    float k0 = -1.0, k1 = 3.0, k2 = 5.0, k3=2.0; 
    int ksize = 4;
    int blocksize = 6;
    float kernelvec[4] = {k0, k1, k2, k3};

    int m = 11, n = 16;
    int m_out = m - ksize + 1;
    int n_out = n - ksize + 1;
    // Create memory aligned buffers
    posix_memalign((void **)&a, 32, m * n * sizeof(float));
    posix_memalign((void **)&op, 32, m_out * n_out * sizeof(float));
    posix_memalign((void **)&op_check, 32, m_out * n_out * sizeof(float));

    // Initialize A
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            a[i*n + j] =  (float)(i*n + j); // ((float)rand()) / ((float)RAND_MAX);
            // printf("%.2f \t", a[i*n + j]);
            // if (j == n-1) printf("\n");
        }
    }

    // Initialize op
    for (int i = 0; i != m_out * n_out; ++i)
    {
        op[i] = 0.0;
        op_check[i] = 0.0;
    }
    naive_conv(m, n, m_out, n_out, ksize, k0, k1, k2, k3, a, op_check);
    t0 = rdtsc();
    // naive_1d_conv(m, k0, k1, k2, a, op_check);
    
    call_kernel(m, n, m_out, n_out, ksize, blocksize, kernelvec, a, op);


    t1 = rdtsc();

    int correct = 1;
    for (int i = 0; i != m_out*n_out; ++i)
    {
        correct &= (fabs(op[i] - op_check[i]) < 1e-13);
    }

    printf("%d\t %lf %d\n", m * n, (2.0 * m * n) / ((double)(t1 - t0) * MAX_FREQ / BASE_FREQ), correct);

    free(a);
    free(op);
    free(op_check);
}