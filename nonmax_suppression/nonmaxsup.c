#include <immintrin.h>
#include <stdio.h>
#include <stdalign.h>

void process_rows_avx2(const float *row0, const float *row1, const float *row2, float *out) {
    // Load 8 floats per row
    __m256 r0 = _mm256_loadu_ps(row0);
    __m256 r1 = _mm256_loadu_ps(row1);

    r0 = _mm256_max_ps(r0, r1);

    // Load next row, vmax again
    r1 = _mm256_loadu_ps(row2);
    r0 = _mm256_max_ps(r0, r1);

    // Shift register in 3s
    __m256i shift3 = _mm256_setr_epi32(1, 2, 0, 4, 5, 3, 7, 6);
    r1 = _mm256_permutevar8x32_ps(r0, shift3);

    r0 = _mm256_max_ps(r0, r1);

    // Shift again
    r1 = _mm256_permutevar8x32_ps(r1, shift3);

    // vmax again
    r0 = _mm256_max_ps(r0, r1);

    // Store every 3rd element
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, r0);

    for (int i = 0; i<2; ++i) {
        out[i] = tmp[3 * i];
    }
}

int main() {
    float row0[8] = {1,2,3,4,5,6,7,8};
    float row1[8] = {8,7,6,5,4,3,2,1};
    float row2[8] = {0,9,2,9,2,9,2,9};
    float out[3] = {0};

    process_rows_avx2(row0, row1, row2, out);

    for (int i = 0; i < 2; ++i)
        printf("out[%d] = %f\n", i, out[i]);

    return 0;
}