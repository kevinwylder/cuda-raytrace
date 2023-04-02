#include "util.c"
#include "math.h"
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

void allocateImage(Image *img, size_t width, size_t height)
{
    img->width = width;
    img->height = height;
    for (int channel = 0; channel < 3; channel++)
    {
        cudaCheck(cudaMallocPitch((void **)&img->data.channel[channel], &img->data.pitch[channel], width, height));
    }

    float phi = 1.6180339887;
    float phi_inv = 1.0 / phi;

    // https://upload.wikimedia.org/wikipedia/commons/a/a4/Dodecahedron_vertices.png
    img->points[0] = {-1, 1, 1};
    img->points[1] = {1, 1, 1};
    img->points[2] = {-1, -1, 1};
    img->points[3] = {1, -1, 1};
    img->points[4] = {-1, 1, -1};
    img->points[5] = {1, 1, -1};
    img->points[6] = {-1, -1, -1};
    img->points[7] = {1, -1, -1};
    img->points[8] = {-phi, 0, phi_inv};
    img->points[9] = {phi, 0, phi_inv};
    img->points[10] = {-phi, 0, -phi_inv};
    img->points[11] = {phi, 0, -phi_inv};
    img->points[12] = {-phi_inv, phi, 0};
    img->points[13] = {phi_inv, phi, 0};
    img->points[14] = {-phi_inv, -phi, 0};
    img->points[15] = {phi_inv, -phi, 0};
    img->points[16] = {0, phi_inv, phi};
    img->points[17] = {0, -phi_inv, phi};
    img->points[18] = {0, phi_inv, -phi};
    img->points[19] = {0, -phi_inv, -phi};

    img->faces[0] = {17, 16, 1, 9, 3};
    img->faces[1] = {1, 16, 0, 12, 13};
    img->faces[2] = {1, 13, 5, 11, 9};
    img->faces[3] = {9, 11, 7, 15, 3};
    img->faces[4] = {17, 3, 15, 14, 2};
    img->faces[5] = {2, 8, 0, 16, 17};
    img->faces[6] = {14, 6, 10, 8, 2};
    img->faces[7] = {4, 12, 0, 8, 10};
    img->faces[8] = {13, 12, 4, 18, 5};
    img->faces[9] = {5, 18, 19, 7, 11};
    img->faces[10] = {6, 14, 15, 7, 19};
    img->faces[11] = {10, 6, 19, 18, 4};
}

typedef struct Ray
{
    Point location;
    Point direction;
} Ray;

__device__ bool collides(Point *a, Point *b, Point *c, Ray *ray)
{
    // https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    Point e1, e2, n, ao, dao;
    sub(&e1, b, a);
    sub(&e2, c, a);
    cross(&n, &e1, &e2);
    sub(&ao, &ray->location, a);
    cross(&dao, &ao, &ray->direction);
    float det = -dot(&ray->direction, &n);
    float invdet = 1.0 / det;
    float u = dot(&e2, &dao) * invdet;
    float v = -dot(&e1, &dao) * invdet;
    float t = dot(&ao, &n) * invdet;
    return (det >= 1e-6 && t >= 0.0 && u >= 0.0 && v >= 0.0 && (u + v) <= 1.0);
}

__global__ void renderPixel(Image *img)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t idx = col * img->width + row;
    float x = (((float)col * 2) / img->width) - 1;
    float y = (((float)row * 2) / img->height) - 1;

    Ray ray = {
        location : {
            x : x * 2.0f,
            y : y * 2.0f,
            z : 10.,
        },
        direction : {
            x : 0,
            y : 0,
            z : -1.,
        }
    };

    img->data.channel[0][idx] = 0;
    img->data.channel[1][idx] = 0;
    img->data.channel[2][idx] = 0;

    for (size_t i = 0; i < 12; i++)
    {
        // check if we collide with a face
        Pentagon face = img->faces[i];
        bool face1 = collides(&img->points[face.a], &img->points[face.b], &img->points[face.c], &ray); // || collides(&img->points[face.a], &img->points[face.c], &img->points[face.b], &ray);
        bool face2 = collides(&img->points[face.a], &img->points[face.c], &img->points[face.d], &ray); // || collides(&img->points[face.a], &img->points[face.d], &img->points[face.c], &ray);
        bool face3 = collides(&img->points[face.a], &img->points[face.d], &img->points[face.e], &ray); // || collides(&img->points[face.a], &img->points[face.e], &img->points[face.d], &ray);
        if (face1)
        {
            img->data.channel[0][idx] = 255;
        }
        if (face2)
        {
            img->data.channel[1][idx] = 255;
        }
        if (face3)
        {
            img->data.channel[2][idx] = 255;
        }
    }
}

void renderImage(Image *img)
{
    Image *deviceImage;
    cudaCheck(cudaMalloc(&deviceImage, sizeof(Image)));
    cudaCheck(cudaMemcpy(deviceImage, img, sizeof(Image), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blocks(img->width / threadsPerBlock.x, img->height / threadsPerBlock.y);
    renderPixel<<<blocks, threadsPerBlock>>>(deviceImage);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(deviceImage));
}

int main()
{
    Image img{
        camera : {0., 0., -2.},
    };
    allocateImage(&img, 4096, 4096);
    debugFace(&img.points[0], &img.faces[8]);
    renderImage(&img);
    writeImageJPEG(&img, "out.jpg");
    return 0;
}
