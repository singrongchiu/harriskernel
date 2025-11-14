#ifndef NONMAXSUP_H
#define NONMAXSUP_H

#include <immintrin.h>
#include <stdio.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif


#define THRESHOLD 1e4f

// -----------------------------------------------------------------------------
// rdtsc() timestamp counter
// -----------------------------------------------------------------------------
static inline unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// -----------------------------------------------------------------------------
// Function Prototypes
// -----------------------------------------------------------------------------

/**
 * Compute max over 3 rows using a single AVX2 register sequence.
 * row0, row1, row2 : pointers to contiguous 8 floats
 * out              : pointer to 2 floats
 */
void process_rows_avx2_onereg(const float *row0,
                              const float *row1,
                              const float *row2,
                              float *out);

/**
 * Max pooling over blocks of 42 floats (14 output values).
 * Each row must contain at least 42 floats starting at the given pointer.
 */
void process_rows_avx2(const float *row0,
                       const float *row1,
                       const float *row2,
                       float *out);

/**
 * Process a full 2D array using AVX2 max-suppression.
 *
 * input  : array of float pointers (height x width)
 * output : array of unsigned char pointers
 * height : number of rows
 * width  : number of columns
 */
void process_array_avx2(float * const *input,
                        unsigned char **output,
                        int height,
                        int width);


#ifdef __cplusplus
}
#endif

#endif // NONMAXSUP_H
