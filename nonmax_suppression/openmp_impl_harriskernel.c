#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "nonmaxsup.h"

#define K 0.04 // Harris detector constant
// #define THRESHOLD 1e4   // Corner response threshold

// Threshold on the final corner map (0–255) when drawing red overlay
#define CORNER_THRESH 250

// gcc -O3 -std=c11 -mavx2 -mfma -fopenmp -o highlevel_omp openmp_impl_harriskernel.c nonmaxsup.c -lm

// -----------------------------------------------------------------------------
// Generic SIMD convolution kernel (your implementation)
// This is a 1D/row-wise kernel primitive over a flat array.
// Now parallelized with OpenMP over rows i (for each r).
// -----------------------------------------------------------------------------
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
    const int UNROLL    = 4;      // 4-way register blocking (loop step is 4*8)

    for (int r = 0; r < ksize; ++r) {

        __m256 kvec = _mm256_set1_ps(k[r]);   // broadcast kernel tap

        // Parallelize over rows i; each thread writes disjoint row in op
        #pragma omp parallel for
        for (int i = 0; i < m_out; ++i) {

            float *outptr = &op[i * n_out];
            float *inptr  = &a[(i + r) * n];

            int j = 0;

            // ---------- UNROLL-way UNROLLED SIMD BLOCK ----------
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

            // ---------- TAIL: scalar ops for leftovers ----------
            for (; j < n_out; ++j) {
                outptr[j] += k[r] * inptr[j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Matrix helpers
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Sobel (existing AVX implementation) + OpenMP over y
// -----------------------------------------------------------------------------
void sobel_avx_simple(float **img, float **Ix, float **Iy, int h, int w)
{
    // Only 4 registers for coefficients
    const __m256 k_m1 = _mm256_set1_ps(-1.0f);
    const __m256 k_p1 = _mm256_set1_ps( 1.0f);
    const __m256 k_m2 = _mm256_set1_ps(-2.0f);
    const __m256 k_p2 = _mm256_set1_ps( 2.0f);

    #pragma omp parallel for
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

// -----------------------------------------------------------------------------
// Gaussian blur via separable 1D kernel using call_kernel (OpenMP)
// 3x3 Gaussian approximated by 4-tap [1 2 1 0]/4 for ksize=4
// -----------------------------------------------------------------------------
void gaussian3(float **src, float **dst, int h, int w) {
    // 4-tap 1D Gaussian-like kernel for call_kernel:
    // effectively [1 2 1]/4, with an extra 0 tap to keep ksize=4.
    float g[4] = {
        1.0f / 4.0f,  // tap 0
        2.0f / 4.0f,  // tap 1
        1.0f / 4.0f,  // tap 2
        0.0f          // tap 3 (dummy)
    };

    // Need at least 3×3 valid region
    if (h < 3 || w < 3) {
        #pragma omp parallel for
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                dst[y][x] = src[y][x];
        return;
    }

    int m      = h;
    int n      = w;
    int ksize  = 4;          // use 4-tap kernel
    int m_out  = m - (ksize - 1);  // = h - 3
    int n_out  = n;          // full width

    // Flatten src into contiguous buffer for call_kernel
    float *flat_in  = (float *)malloc(m * n * sizeof(float));
    float *flat_out = (float *)malloc(m_out * n_out * sizeof(float));
    if (!flat_in || !flat_out) {
        fprintf(stderr, "gaussian3: malloc failed\n");
        free(flat_in);
        free(flat_out);
        return;
    }

    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            flat_in[y * n + x] = src[y][x];
        }
    }

    // Vertical 1D convolution with 4-tap kernel (last tap = 0)
    call_kernel(
        m, n,
        m_out, n_out,
        ksize,
        /*blocksize*/ 1,
        g,        // 4-tap kernel
        flat_in,  // input (flattened)
        flat_out  // output (flattened, size (h-3) x w)
    );

    // Temporary 2D buffer to hold vertically-blurred result
    float **tmp = alloc_matrix(h, w);

    // Map flat_out back into tmp:
    // call_kernel computes:
    //   out[i, x] = Σ_r g[r] * in[i+r, x], r=0..3
    // but g[3] = 0, so effective rows are i, i+1, i+2.
    // Their "center" is row (i+1), so write into y = i+1.
    #pragma omp parallel for
    for (int i = 0; i < m_out; ++i) {  // i: 0..h-4
        int y = i + 1;                 // y: 1..h-3
        for (int x = 0; x < w; ++x) {
            tmp[y][x] = flat_out[i * n_out + x];
        }
    }

    // Zero borders of tmp (unused / padded)
    for (int x = 0; x < w; ++x) {
        tmp[0][x]     = 0.0f;
        tmp[h - 1][x] = 0.0f;
    }
    for (int y = 0; y < h; ++y) {
        tmp[y][0]     = 0.0f;
        tmp[y][w - 1] = 0.0f;
    }

    free(flat_in);
    free(flat_out);

    // Horizontal pass: same 1D [1 2 1]/4 kernel, scalar
    #pragma omp parallel for
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float left   = tmp[y][x - 1];
            float center = tmp[y][x];
            float right  = tmp[y][x + 1];

            dst[y][x] = g[0] * left + g[1] * center + g[2] * right;
        }
    }

    // Set borders of dst to 0 (Harris only uses interior anyway)
    for (int x = 0; x < w; ++x) {
        dst[0][x]     = 0.0f;
        dst[h - 1][x] = 0.0f;
    }
    for (int y = 0; y < h; ++y) {
        dst[y][0]     = 0.0f;
        dst[y][w - 1] = 0.0f;
    }

    free_matrix(tmp, h);
}

// -----------------------------------------------------------------------------
// Harris response + nonmax
// -----------------------------------------------------------------------------
void harris_response(float **Ix, float **Iy, float **R, int h, int w) {
    float **Ix2 = alloc_matrix(h, w);
    float **Iy2 = alloc_matrix(h, w);
    float **Ixy = alloc_matrix(h, w);

    // Compute products
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    process_array_avx2(R, out, h, w);   // from nonmaxsup.c/h

    free_matrix(Ix, h);
    free_matrix(Iy, h);
    free_matrix(R, h);
}

// -----------------------------------------------------------------------------
// IO helpers
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main() {
    int w, h;
    float **gray_img = load_jpg_as_grayscale_f32("chessboard.jpg", &h, &w);
    if (!gray_img) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

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

    printf("RDTSC Base Cycles Taken for HARRIS CORNER: %llu\n", sum);

    // -----------------------------------------------------------------
    // Build color overlay: grayscale background + red pixels at corners
    // -----------------------------------------------------------------
    unsigned char *outbuf = malloc(h * w * 3);
    if (!outbuf) {
        fprintf(stderr, "Failed to allocate outbuf\n");
        return 1;
    }

    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;

            // grayscale from gray_img (0–255)
            float g = gray_img[y][x];
            if (g < 0.0f)   g = 0.0f;
            if (g > 255.0f) g = 255.0f;
            unsigned char gv = (unsigned char)g;

            // default background = grayscale
            outbuf[idx + 0] = gv;  // R
            outbuf[idx + 1] = gv;  // G
            outbuf[idx + 2] = gv;  // B

            // corners[y][x] is from process_array_avx2 (likely 0 or 255)
            if (corners[y][x] > CORNER_THRESH) {
                // mark as red
                outbuf[idx + 0] = 255;  // R
                outbuf[idx + 1] = 0;    // G
                outbuf[idx + 2] = 0;    // B
            }
        }
    }

    // Write to JPEG (3 channels, quality 95)
    stbi_write_jpg("corners_overlay.jpg", w, h, 3, outbuf, 95);
    printf("Done. Saved corners_overlay.jpg\n");

    // Free memory
    for (int i = 0; i < h; i++) {
        free(gray_img[i]);
        free(corners[i]);
    }
    free(gray_img);
    free(corners);
    free(outbuf);

    printf("Done.\n");
    return 0;
}
