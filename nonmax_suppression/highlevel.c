#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define K 0.04   // Harris detector constant
#define THRESHOLD 1e3   // Corner response threshold

// gcc -O2 -std=c11 -mavx2 -o highlevel highlevel.c -lm

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


void sobel(float **img, float **Ix, float **Iy, int h, int w) {
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gy[3][3] = {
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1}
    };

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float sx = 0, sy = 0;
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++) {
                    sx += img[y + j - 1][x + i - 1] * gx[j][i];
                    sy += img[y + j - 1][x + i - 1] * gy[j][i];
                }
            Ix[y][x] = sx;
            Iy[y][x] = sy;
        }
    }
}


void gaussian3(float **src, float **dst, int h, int w) {
    float g[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    float norm = 16.0;

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float sum = 0;
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++)
                    sum += src[y + j - 1][x + i - 1] * g[j][i];

            dst[y][x] = sum / norm;
        }
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

    sobel(image, Ix, Iy, h, w);
    harris_response(Ix, Iy, R, h, w);
    nonmax(R, out, h, w);

    free_matrix(Ix, h);
    free_matrix(Iy, h);
    free_matrix(R, h);
}

// Converts RGB to grayscale using luminance formula
unsigned char rgb_to_gray(unsigned char r, unsigned char g, unsigned char b) {
    return (unsigned char)(0.299*r + 0.587*g + 0.114*b);
}

// Load JPG → convert to grayscale float matrix
float **load_jpg_as_grayscale_f32(const char *filename, int *h, int *w) {
    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    if (!data) {
        printf("Error loading %s\n", filename);
        return NULL;
    }

    *w = width;
    *h = height;

    // Allocate 2D float image
    float **img = malloc(height * sizeof(float*));
    for (int y = 0; y < height; y++)
        img[y] = malloc(width * sizeof(float));

    // Convert each pixel to grayscale
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            unsigned char R = data[idx];
            unsigned char G = data[idx + 1];
            unsigned char B = data[idx + 2];
            img[y][x] = rgb_to_gray(R, G, B);
        }
    }

    stbi_image_free(data);
    return img;
}

int main() {
    int w, h;
    float **gray_img = load_jpg_as_grayscale_f32("chessboard.jpg", &h, &w);

    unsigned char **corners = malloc(h * sizeof(unsigned char*));
    for (int i = 0; i < h; i++)
        corners[i] = calloc(w, 1);

    harris_corner_detector(gray_img, corners, h, w);

    // start with gray
    unsigned char *outbuf = malloc(h*w*3);
    for (int y=0;y<h;y++) {
        for (int x=0;x<w;x++) {
            int idx = (y*w + x)*3;
            unsigned char gray = (unsigned char)gray_img[y][x];
            outbuf[idx+0] = gray; // R
            outbuf[idx+1] = gray; // G
            outbuf[idx+2] = gray; // B
        }
    }

    // Overlay corners in red
    for (int y=0;y<h;y++) {
        for (int x=0;x<w;x++) {
            if (corners[y][x] > 0) {
                int idx = (y*w + x)*3;
                outbuf[idx+0] = 255; // Red
                outbuf[idx+1] = 0;   // Green
                outbuf[idx+2] = 0;   // Blue
            }
        }
    }

    stbi_write_jpg("corners_red.jpg",w,h,3,outbuf,95);
    printf("Saved corners_red.jpg\n");

    for (int i=0;i<h;i++) {
        free(gray_img[i]);
        free(corners[i]);
    }
    free(gray_img);
    free(corners);
    free(outbuf);

    printf("Done.\n");
}

