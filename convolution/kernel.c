#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void call_kernel(
    int m, int n,
    int m_out, int n_out,
    int ksize,
    int blocksize,          // ignored in SIMD but kept for API
    float *restrict k,
    float *restrict a,
    float *restrict op)
{
    // initialize output
    for (int i = 0; i < m_out * n_out; ++i)
        op[i] = 0.0f;

    const int VEC_WIDTH = 8;      // 8 floats per AVX2 vector
    const int UNROLL    = 5;      // 4-way register blocking

    for (int r = 0; r < ksize; ++r) {

        __m256 kvec = _mm256_set1_ps(k[r]);   // broadcast kernel tap

        for (int i = 0; i < m_out; ++i) {

            float *outptr = &op[i * n_out];
            float *inptr  = &a[(i + r) * n];

            int j = 0;

            // ---------- ${UNROLL}-WAY UNROLLED SIMD BLOCK ----------
            for (; j + UNROLL * VEC_WIDTH - 1 < n_out; j += UNROLL * VEC_WIDTH) {

                __m256 L0 = _mm256_loadu_ps(inptr  + j + 0*VEC_WIDTH);
                __m256 A0 = _mm256_loadu_ps(outptr + j + 0*VEC_WIDTH);
                A0 = _mm256_fmadd_ps(L0, kvec, A0);
                _mm256_storeu_ps(outptr + j + 0*VEC_WIDTH, A0);

                __m256 L1 = _mm256_loadu_ps(inptr  + j + 1*VEC_WIDTH);
                __m256 A1 = _mm256_loadu_ps(outptr + j + 1*VEC_WIDTH);
                A1 = _mm256_fmadd_ps(L1, kvec, A1);
                _mm256_storeu_ps(outptr + j + 1*VEC_WIDTH, A1);

                __m256 L2 = _mm256_loadu_ps(inptr  + j + 2*VEC_WIDTH);
                __m256 A2 = _mm256_loadu_ps(outptr + j + 2*VEC_WIDTH);
                A2 = _mm256_fmadd_ps(L2, kvec, A2);
                _mm256_storeu_ps(outptr + j + 2*VEC_WIDTH, A2);

                __m256 L3 = _mm256_loadu_ps(inptr  + j + 3*VEC_WIDTH);
                __m256 A3 = _mm256_loadu_ps(outptr + j + 3*VEC_WIDTH);
                A3 = _mm256_fmadd_ps(L3, kvec, A3);
                _mm256_storeu_ps(outptr + j + 3*VEC_WIDTH, A3);
            }

            // ---------- FALLBACK: single full SIMD block ----------
            // for (; j + VEC_WIDTH - 1 < n_out; j += VEC_WIDTH) {

            //     __m256 L = _mm256_loadu_ps(inptr + j);
            //     __m256 A = _mm256_loadu_ps(outptr + j);
            //     A = _mm256_fmadd_ps(L, kvec, A);
            //     _mm256_storeu_ps(outptr + j, A);
            // }

            // ---------- TAIL: scalar ops for leftovers ----------
            for (; j < n_out; ++j) {
                outptr[j] += k[r] * inptr[j];
            }
        }
    }
}


// void conv_1d(
//     int idx0,
//     int idx1, 
//     int idx2,
//     int idx3,
//     float k0,
//     float k1,
//     float k2,
//     float *restrict a,
//     float *restrict op_check)
// {
//     printf("\n here \n");
// }

// void call_kernel(
//     int m,
//     int k, // kernel size = 3
//     float k0,
//     float k1,
//     float k2,
//     float *restrict a,
//     float *restrict op_check)
// {
//     printf("\n here \n");
// }
