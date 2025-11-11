#include <immintrin.h>
#include <stdio.h>
#include <stdalign.h>
// gcc -O2 -std=c11 -mavx2 -o nonmaxsup nonmaxsup.c

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void process_rows_avx2(const float *row0, const float *row1, const float *row2, float *out) {
    // load first two rows
    // r0r1 r2r3 r4r5 r6r7 r8r9 r10r11 r12r13 r14r15
    __m256 r0 = _mm256_loadu_ps(row0);
    __m256 r1 = _mm256_loadu_ps(row1);

    __m256 r2 = _mm256_loadu_ps(row0 + 6);
    __m256 r3 = _mm256_loadu_ps(row1 + 6);

    __m256 r4 = _mm256_loadu_ps(row0 + 12);
    __m256 r5 = _mm256_loadu_ps(row1 + 12);

    __m256 r6 = _mm256_loadu_ps(row0 + 18);
    __m256 r7 = _mm256_loadu_ps(row1 + 18);

    __m256 r8 = _mm256_loadu_ps(row0 + 24);
    __m256 r9 = _mm256_loadu_ps(row1 + 24);

    __m256 r10 = _mm256_loadu_ps(row0 + 30);
    __m256 r11 = _mm256_loadu_ps(row1 + 30);

    __m256 r12 = _mm256_loadu_ps(row0 + 36);
    __m256 r13 = _mm256_loadu_ps(row1 + 36);

    __m256 r14 = _mm256_loadu_ps(row0 + 42);
    __m256 r15 = _mm256_loadu_ps(row1 + 42);

    r0 = _mm256_max_ps(r0, r1);
    r2 = _mm256_max_ps(r2, r3);
    r4 = _mm256_max_ps(r4, r5);
    r6 = _mm256_max_ps(r6, r7);
    r8 = _mm256_max_ps(r8, r9);
    r10 = _mm256_max_ps(r10, r11);
    r12 = _mm256_max_ps(r12, r13);
    r14 = _mm256_max_ps(r14, r15);

    // load next row, vmax again
    r1 = _mm256_loadu_ps(row2);
    r3 = _mm256_loadu_ps(row2 + 6);
    r5 = _mm256_loadu_ps(row2 + 12);
    r7 = _mm256_loadu_ps(row2 + 18);
    r9 = _mm256_loadu_ps(row2 + 24);
    r11 = _mm256_loadu_ps(row2 + 30);
    r13 = _mm256_loadu_ps(row2 + 36);
    r15 = _mm256_loadu_ps(row2 + 42);

    r0 = _mm256_max_ps(r0, r1);
    r2 = _mm256_max_ps(r2, r3);
    r4 = _mm256_max_ps(r4, r5);
    r6 = _mm256_max_ps(r6, r7);
    r8 = _mm256_max_ps(r8, r9);
    r10 = _mm256_max_ps(r10, r11);
    r12 = _mm256_max_ps(r12, r13);
    r14 = _mm256_max_ps(r14, r15);

    // shift register in 3s
    __m256i shift3 = _mm256_setr_epi32(1, 2, 0, 4, 5, 3, 7, 6);
    r1 = _mm256_permutevar8x32_ps(r0, shift3);
    r3 = _mm256_permutevar8x32_ps(r2, shift3);
    r5 = _mm256_permutevar8x32_ps(r4, shift3);
    r7 = _mm256_permutevar8x32_ps(r6, shift3);
    r9 = _mm256_permutevar8x32_ps(r8, shift3);
    r11 = _mm256_permutevar8x32_ps(r10, shift3);
    r13 = _mm256_permutevar8x32_ps(r12, shift3);
    r15 = _mm256_permutevar8x32_ps(r14, shift3);

    r0 = _mm256_max_ps(r0, r1);
    r2 = _mm256_max_ps(r2, r3);
    r4 = _mm256_max_ps(r4, r5);
    r6 = _mm256_max_ps(r6, r7);
    r8 = _mm256_max_ps(r8, r9);
    r10 = _mm256_max_ps(r10, r11);
    r12 = _mm256_max_ps(r12, r13);
    r14 = _mm256_max_ps(r14, r15);

    // shift again
    r1 = _mm256_permutevar8x32_ps(r1, shift3);
    r3 = _mm256_permutevar8x32_ps(r3, shift3);
    r5 = _mm256_permutevar8x32_ps(r5, shift3);
    r7 = _mm256_permutevar8x32_ps(r7, shift3);
    r9 = _mm256_permutevar8x32_ps(r9, shift3);
    r11 = _mm256_permutevar8x32_ps(r11, shift3);
    r13 = _mm256_permutevar8x32_ps(r13, shift3);
    r15 = _mm256_permutevar8x32_ps(r15, shift3);

    // vmax again
    r0 = _mm256_max_ps(r0, r1);
    r2 = _mm256_max_ps(r2, r3);
    r4 = _mm256_max_ps(r4, r5);
    r6 = _mm256_max_ps(r6, r7);
    r8 = _mm256_max_ps(r8, r9);
    r10 = _mm256_max_ps(r10, r11);
    r12 = _mm256_max_ps(r12, r13);
    r14 = _mm256_max_ps(r14, r15);

    // Store every 3rd element
    alignas(32) float tmp[8][8];
    for (int i = 0; i < 8; ++i) {
        _mm256_store_ps(tmp[i], (i==0)?r0:(i==1)?r2:(i==2)?r4:(i==3)?r6:(i==4)?r8:(i==5)?r10:(i==6)?r12:r14);
    }
    // _mm256_store_ps(tmp, r0);

    for (int i = 0; i<8; ++i) {
        for (int j = 0; j < 2; ++j) {
            out[i * 2 + j] = tmp[i][j * 3];
        }
    }
}

int main() {
    float row2[48] = {
        1,2,3,4,5,6,7,8,
        9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,
        25,26,27,28,29,30,31,32,
        33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48
    };

    float row1[48] = {
        48,47,46,45,44,43,42,41,
        40,39,38,37,36,35,34,33,
        32,31,30,29,28,27,26,25,
        24,23,22,21,20,19,18,17,
        16,15,14,13,12,11,10,9,
        8,7,6,5,4,3,2,1
    };

    float row0[48] = {
        0,9,2,9,2,9,2,9,
        0,9,2,9,2,9,2,9,
        0,9,2,9,2,9,2,9,
        0,9,2,9,2,9,2,9,
        0,9,2,9,2,9,2,9,
        0,9,2,9,2,9,2,9
    };
    float out[16] = {0};

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    st = rdtsc();
    process_rows_avx2(row0, row1, row2, out);
    et = rdtsc();
    sum += (et-st);

    printf("RDTSC Base Cycles Taken for INT_MUL: %llu\n\r",sum);

    for (int i = 0; i < 16; ++i)
        printf("out[%d] = %f\n", i, out[i]);

    return 0;
}