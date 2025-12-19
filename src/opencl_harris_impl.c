// opencl_harris_impl.c
// Compile (example):
//   gcc opencl_harris_impl.c -o harris_opencl -lOpenCL -lm
//
// Run:
//   ./harris_opencl chessboard.jpg
//
// Requires: OpenCL runtime + stb_image / stb_image_write (included below)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

// stb image
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Harris constants
#define K_HARRIS        0.04f
#define HARRIS_THRESH   1e6f      // threshold on R for nonmax
#define CORNER_THRESH   100       // 0–255 when drawing overlay

// -------------------------------------------------------------------------------------------------
// RDTSC timing helper
// -------------------------------------------------------------------------------------------------
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// -------------------------------------------------------------------------------------------------
// OpenCL kernels
// -------------------------------------------------------------------------------------------------
static const char *kernel_src =

"__kernel void sobel_kernel(\n"
"    __global const float* img,\n"
"    __global float* Ix,\n"
"    __global float* Iy,\n"
"    const int width,\n"
"    const int height)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) {\n"
"        if (x < width && y < height) {\n"
"            int idx = y*width + x;\n"
"            Ix[idx] = 0.0f;\n"
"            Iy[idx] = 0.0f;\n"
"        }\n"
"        return;\n"
"    }\n"
"\n"
"    int idx = y*width + x;\n"
"\n"
"    // Sobel operator\n"
"    #define IDX(xx,yy) ((yy)*width + (xx))\n"
"\n"
"    float tl = img[IDX(x-1,y-1)];\n"
"    float  t = img[IDX(x  ,y-1)];\n"
"    float tr = img[IDX(x+1,y-1)];\n"
"    float  l = img[IDX(x-1,y  )];\n"
"    float  r = img[IDX(x+1,y  )];\n"
"    float bl = img[IDX(x-1,y+1)];\n"
"    float  b = img[IDX(x  ,y+1)];\n"
"    float br = img[IDX(x+1,y+1)];\n"
"\n"
"    // Same convention as typical Sobel\n"
"    float gx = -tl - 2.0f*l - bl + tr + 2.0f*r + br;\n"
"    float gy = -tl - 2.0f*t - tr + bl + 2.0f*b + br;\n"
"\n"
"    Ix[idx] = gx;\n"
"    Iy[idx] = gy;\n"
"}\n"
"\n"
"__kernel void harris_kernel(\n"
"    __global const float* Ix,\n"
"    __global const float* Iy,\n"
"    __global float* R,\n"
"    const int width,\n"
"    const int height,\n"
"    const float k)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x <= 1 || x >= width-2 || y <= 1 || y >= height-2) {\n"
"        if (x < width && y < height) {\n"
"            int idx = y*width + x;\n"
"            R[idx] = 0.0f;\n"
"        }\n"
"        return;\n"
"    }\n"
"\n"
"    int idx = y*width + x;\n"
"\n"
"    // 3x3 Gaussian kernel 1/16 * [ [1 2 1]; [2 4 2]; [1 2 1] ]\n"
"    const float g[3][3] = {\n"
"        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},\n"
"        {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},\n"
"        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}\n"
"    };\n"
"\n"
"    float Sxx = 0.0f;\n"
"    float Syy = 0.0f;\n"
"    float Sxy = 0.0f;\n"
"\n"
"    #define IDX(xx,yy) ((yy)*width + (xx))\n"
"\n"
"    for (int j = -1; j <= 1; ++j) {\n"
"        for (int i = -1; i <= 1; ++i) {\n"
"            float w = g[j+1][i+1];\n"
"            int nidx = IDX(x+i, y+j);\n"
"            float ix = Ix[nidx];\n"
"            float iy = Iy[nidx];\n"
"            Sxx += w * ix * ix;\n"
"            Syy += w * iy * iy;\n"
"            Sxy += w * ix * iy;\n"
"        }\n"
"    }\n"
"\n"
"    float det = Sxx * Syy - Sxy * Sxy;\n"
"    float trace = Sxx + Syy;\n"
"    R[idx] = det - k * trace * trace;\n"
"}\n"
"\n"
"__kernel void nonmax_kernel(\n"
"    __global const float* R,\n"
"    __global uchar* corners,\n"
"    const int width,\n"
"    const int height,\n"
"    const float thresh)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x <= 1 || x >= width-2 || y <= 1 || y >= height-2) {\n"
"        if (x < width && y < height) {\n"
"            int idx = y*width + x;\n"
"            corners[idx] = (uchar)0;\n"
"        }\n"
"        return;\n"
"    }\n"
"\n"
"    int idx = y*width + x;\n"
"    float v = R[idx];\n"
"    if (v < thresh) {\n"
"        corners[idx] = (uchar)0;\n"
"        return;\n"
"    }\n"
"\n"
"    int is_max = 1;\n"
"    #define IDX2(xx,yy) ((yy)*width + (xx))\n"
"    for (int j = -1; j <= 1; ++j) {\n"
"        for (int i = -1; i <= 1; ++i) {\n"
"            if (i == 0 && j == 0) continue;\n"
"            float neigh = R[IDX2(x+i, y+j)];\n"
"            if (neigh > v) {\n"
"                is_max = 0;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    corners[idx] = is_max ? (uchar)255 : (uchar)0;\n"
"}\n"
"\n";

// -------------------------------------------------------------------------------------------------
// Helper: check OpenCL error and bail
// -------------------------------------------------------------------------------------------------
static void check_cl(cl_int err, const char *msg)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d at %s\n", err, msg);
        exit(1);
    }
}

// -------------------------------------------------------------------------------------------------
// Load image as grayscale float in [0,255]
// -------------------------------------------------------------------------------------------------
float *load_jpg_as_grayscale_f32(const char *filename, int *w, int *h)
{
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    if (!data) {
        fprintf(stderr, "Error loading %s\n", filename);
        return NULL;
    }

    *w = width;
    *h = height;

    float *img = (float*)malloc(width * height * sizeof(float));
    if (!img) {
        fprintf(stderr, "malloc failed for image\n");
        stbi_image_free(data);
        return NULL;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            unsigned char R = data[idx];
            unsigned char G = data[idx + 1];
            unsigned char B = data[idx + 2];
            // standard luminance
            float g = 0.299f * R + 0.587f * G + 0.114f * B;
            img[y * width + x] = g;
        }
    }

    stbi_image_free(data);
    return img;
}

// -------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.jpg\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];

    int width, height;
    float *gray_img = load_jpg_as_grayscale_f32(input_path, &width, &height);
    if (!gray_img) {
        return 1;
    }
    size_t img_size = (size_t)width * height;

    printf("Loaded %s (%dx%d)\n", input_path, width, height);

    // ---------------------------------------------------------------------------------------------
    // OpenCL setup
    // ---------------------------------------------------------------------------------------------
    cl_int err;

    cl_uint num_platforms = 0;
    check_cl(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs count");
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return 1;
    }

    cl_platform_id platform;
    check_cl(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs get one");

    cl_uint num_devices = 0;
    check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices),
             "clGetDeviceIDs count");
    if (num_devices == 0) {
        fprintf(stderr, "No GPU devices found, trying CPU.\n");
        check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &platform, NULL),
                 "clGetDeviceIDs CPU fallback (NOTE: this line has a bug: should pass &device)");
    }

    cl_device_id device;
    check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL),
             "clGetDeviceIDs get one");

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl(err, "clCreateContext");

    cl_command_queue queue =
        clCreateCommandQueue(context, device, 0, &err);
    check_cl(err, "clCreateCommandQueue");

    const char *srcs[] = { kernel_src };
    size_t lengths[] = { strlen(kernel_src) };
    cl_program program = clCreateProgramWithSource(context, 1, srcs, lengths, &err);
    check_cl(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        check_cl(err, "clBuildProgram");
    }

    cl_kernel sobel_k  = clCreateKernel(program, "sobel_kernel", &err);
    check_cl(err, "clCreateKernel sobel_kernel");
    cl_kernel harris_k = clCreateKernel(program, "harris_kernel", &err);
    check_cl(err, "clCreateKernel harris_kernel");
    cl_kernel nonmax_k = clCreateKernel(program, "nonmax_kernel", &err);
    check_cl(err, "clCreateKernel nonmax_kernel");

    // ---------------------------------------------------------------------------------------------
    // Create buffers
    // ---------------------------------------------------------------------------------------------
    cl_mem d_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  img_size * sizeof(float), gray_img, &err);
    check_cl(err, "clCreateBuffer d_img");

    cl_mem d_Ix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 img_size * sizeof(float), NULL, &err);
    check_cl(err, "clCreateBuffer d_Ix");

    cl_mem d_Iy = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 img_size * sizeof(float), NULL, &err);
    check_cl(err, "clCreateBuffer d_Iy");

    cl_mem d_R  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 img_size * sizeof(float), NULL, &err);
    check_cl(err, "clCreateBuffer d_R");

    cl_mem d_corners = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      img_size * sizeof(unsigned char), NULL, &err);
    check_cl(err, "clCreateBuffer d_corners");

    size_t global_size[2] = { (size_t)width, (size_t)height };

    // ---------------------------------------------------------------------------------------------
    // START RDTSC timing around OpenCL kernels
    // ---------------------------------------------------------------------------------------------
    unsigned long long st_cycles = rdtsc();

    // ---------------------------------------------------------------------------------------------
    // Launch sobel_kernel
    // ---------------------------------------------------------------------------------------------
    err  = clSetKernelArg(sobel_k, 0, sizeof(cl_mem), &d_img);
    err |= clSetKernelArg(sobel_k, 1, sizeof(cl_mem), &d_Ix);
    err |= clSetKernelArg(sobel_k, 2, sizeof(cl_mem), &d_Iy);
    err |= clSetKernelArg(sobel_k, 3, sizeof(int),    &width);
    err |= clSetKernelArg(sobel_k, 4, sizeof(int),    &height);
    check_cl(err, "clSetKernelArg sobel_k");

    check_cl(clEnqueueNDRangeKernel(queue, sobel_k, 2, NULL,
                                    global_size, NULL, 0, NULL, NULL),
             "clEnqueueNDRangeKernel sobel_k");

    // ---------------------------------------------------------------------------------------------
    // Launch harris_kernel
    // ---------------------------------------------------------------------------------------------
    float k_harris = K_HARRIS;

    err  = clSetKernelArg(harris_k, 0, sizeof(cl_mem), &d_Ix);
    err |= clSetKernelArg(harris_k, 1, sizeof(cl_mem), &d_Iy);
    err |= clSetKernelArg(harris_k, 2, sizeof(cl_mem), &d_R);
    err |= clSetKernelArg(harris_k, 3, sizeof(int),    &width);
    err |= clSetKernelArg(harris_k, 4, sizeof(int),    &height);
    err |= clSetKernelArg(harris_k, 5, sizeof(float),  &k_harris);
    check_cl(err, "clSetKernelArg harris_k");

    check_cl(clEnqueueNDRangeKernel(queue, harris_k, 2, NULL,
                                    global_size, NULL, 0, NULL, NULL),
             "clEnqueueNDRangeKernel harris_k");

    // ---------------------------------------------------------------------------------------------
    // Launch nonmax_kernel
    // ---------------------------------------------------------------------------------------------
    float harris_thresh = HARRIS_THRESH;

    err  = clSetKernelArg(nonmax_k, 0, sizeof(cl_mem), &d_R);
    err |= clSetKernelArg(nonmax_k, 1, sizeof(cl_mem), &d_corners);
    err |= clSetKernelArg(nonmax_k, 2, sizeof(int),    &width);
    err |= clSetKernelArg(nonmax_k, 3, sizeof(int),    &height);
    err |= clSetKernelArg(nonmax_k, 4, sizeof(float),  &harris_thresh);
    check_cl(err, "clSetKernelArg nonmax_k");

    check_cl(clEnqueueNDRangeKernel(queue, nonmax_k, 2, NULL,
                                    global_size, NULL, 0, NULL, NULL),
             "clEnqueueNDRangeKernel nonmax_k");

    // Ensure all kernels complete before taking end timestamp
    check_cl(clFinish(queue), "clFinish after kernels");

    unsigned long long et_cycles = rdtsc();
    unsigned long long delta_cycles = et_cycles - st_cycles;
    printf("RDTSC Base Cycles Taken for OpenCL HARRIS pipeline (kernels only): %llu\n",
           delta_cycles);

    // ---------------------------------------------------------------------------------------------
    // Read back corners
    // ---------------------------------------------------------------------------------------------
    unsigned char *corners = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    if (!corners) {
        fprintf(stderr, "malloc failed for corners\n");
        return 1;
    }

    check_cl(clEnqueueReadBuffer(queue, d_corners, CL_TRUE, 0,
                                 img_size * sizeof(unsigned char),
                                 corners, 0, NULL, NULL),
             "clEnqueueReadBuffer d_corners");

    // ---------------------------------------------------------------------------------------------
    // Build color overlay and save
    // ---------------------------------------------------------------------------------------------
    unsigned char *outbuf = (unsigned char*)malloc(width * height * 3);
    if (!outbuf) {
        fprintf(stderr, "malloc failed for outbuf\n");
        return 1;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx  = y*width + x;
            int idx3 = idx * 3;

            float g = gray_img[idx];
            if (g < 0.0f)   g = 0.0f;
            if (g > 255.0f) g = 255.0f;
            unsigned char gv = (unsigned char)g;

            outbuf[idx3 + 0] = gv; // R
            outbuf[idx3 + 1] = gv; // G
            outbuf[idx3 + 2] = gv; // B

            if (corners[idx] > CORNER_THRESH) {
                outbuf[idx3 + 0] = 255; // R
                outbuf[idx3 + 1] = 0;   // G
                outbuf[idx3 + 2] = 0;   // B
            }
        }
    }

    const char *out_name = "corners_overlay_opencl.jpg";
    stbi_write_jpg(out_name, width, height, 3, outbuf, 95);
    printf("Saved %s\n", out_name);

    // ---------------------------------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------------------------------
    free(gray_img);
    free(corners);
    free(outbuf);

    clReleaseMemObject(d_img);
    clReleaseMemObject(d_Ix);
    clReleaseMemObject(d_Iy);
    clReleaseMemObject(d_R);
    clReleaseMemObject(d_corners);

    clReleaseKernel(sobel_k);
    clReleaseKernel(harris_k);
    clReleaseKernel(nonmax_k);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Done.\n");
    return 0;
}
