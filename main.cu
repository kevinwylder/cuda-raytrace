#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nvjpeg.h>
#include <fcntl.h>

#include "scene.cu"

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

void writeImageJPEG(Scene *img, const char *out)
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

__global__ void renderPixel(Scene *img)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t idx = col * img->width + row;
    float x = (((float)col * 2) / img->width) - 1;
    float y = (((float)row * 2) / img->height) - 1;
    assignColor(img, idx, x, y);
}

void renderImage(Scene *img)
{
    Scene *deviceImage;
    cudaCheck(cudaMalloc(&deviceImage, sizeof(Scene)));
    cudaCheck(cudaMemcpy(deviceImage, img, sizeof(Scene), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blocks(img->width / threadsPerBlock.x, img->height / threadsPerBlock.y);
    renderPixel<<<blocks, threadsPerBlock>>>(deviceImage);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(deviceImage));
}

int main()
{
    Scene img{
        camera : {0., 0., 6.},
        macro : .4,
        rotX : .049,
        rotY : .102,
        band : 0.004,
        reflectivity : .87,
        reflections : 30,
        width : 4096,
        height : 4096,
    };

    // allocate image
    for (int channel = 0; channel < 3; channel++)
    {
        cudaCheck(cudaMallocPitch(
            (void **)&img.data.channel[channel],
            &img.data.pitch[channel],
            img.width,
            img.height));
    }
    initializeScene(&img, 4096, 4096);
    renderImage(&img);
    writeImageJPEG(&img, "out.jpg");
    return 0;
}
