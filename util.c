#include <nvjpeg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include "image.h"

void jpegAssert(nvjpegStatus_t status, const char *file, int line)
{
    const char *name = NULL;
    switch (status)
    {
    case NVJPEG_STATUS_SUCCESS:
        return;
    case NVJPEG_STATUS_INVALID_PARAMETER:
        name = "invalid parameter";
        break;
    }
    if (name)
    {
        printf("nvjpeg runtime assert %s:%d - %s\n", file, line, name);
    }
    else
    {
        printf("nvjpeg runtime assert %s:%d - code %d\n", file, line, status);
    }
    exit(1);
}

void cudaAssert(cudaError_t status, const char *file, int line)
{
    if (status == cudaSuccess)
    {
        return;
    }
    printf("cuda runtime assert %d (%s) in %s:%d", status, cudaGetErrorName(status), file, line);
    exit(1);
}

#define nvjpegCheck(call)                       \
    {                                           \
        jpegAssert((call), __FILE__, __LINE__); \
    }

#define cudaCheck(call)                         \
    {                                           \
        cudaAssert((call), __FILE__, __LINE__); \
    }

void writeImageJPEG(Image *img, const char *out)
{
    nvjpegHandle_t handle;
    nvjpegEncoderState_t state;
    nvjpegEncoderParams_t params;
    // default stream
    cudaStream_t stream = 0;

    nvjpegCheck(nvjpegCreateSimple(&handle));
    nvjpegCheck(nvjpegEncoderStateCreate(handle, &state, stream));
    nvjpegCheck(nvjpegEncoderParamsCreate(handle, &params, stream));
    nvjpegCheck(nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_444, stream));

    nvjpegCheck(nvjpegEncodeImage(
        handle, state, params,
        &img->data, NVJPEG_INPUT_RGB,
        img->width, img->height,
        stream));

    size_t length;
    // get size of output image
    nvjpegCheck(nvjpegEncodeRetrieveBitstream(handle, state, NULL, &length, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    unsigned char *jpegRaw;
    cudaCheck(cudaHostAlloc((void **)&jpegRaw, length, cudaHostAllocMapped));

    // copy the image into the buffer
    nvjpegCheck(nvjpegEncodeRetrieveBitstream(handle, state, jpegRaw, &length, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    // copy buffer to file
    // TODO mmap + fallocate? check errors?
    int fd = open(out, O_CREAT | O_RDWR | O_TRUNC, 0666);
    write(fd, jpegRaw, length);
}

void debugFace(Point *points, Pentagon *face)
{
    printf(
        "(%0.6f, %0.6f, %0.6f)\n(%0.6f, %0.6f, %0.6f)\n(%0.6f, %0.6f, %0.6f)\n(%0.6f, %0.6f, %0.6f)\n(%0.6f, %0.6f, %0.6f)\n",
        points[face->a].x, points[face->a].y, points[face->a].z,
        points[face->b].x, points[face->b].y, points[face->b].z,
        points[face->c].x, points[face->c].y, points[face->c].z,
        points[face->d].x, points[face->d].y, points[face->d].z,
        points[face->e].x, points[face->e].y, points[face->e].z);
}