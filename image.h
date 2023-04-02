#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <nvjpeg.h>
#include "math.h"

typedef struct Pentagon
{
    size_t a;
    size_t b;
    size_t c;
    size_t d;
    size_t e;
} Pentagon;

typedef struct Image
{
    float camera[3];
    size_t width;
    size_t height;
    Point points[20];
    Pentagon faces[12];

    nvjpegImage_t data;
} Image;

#endif