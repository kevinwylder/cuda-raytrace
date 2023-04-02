#ifndef __MATH_H__
#define __MATH_H__

typedef struct Point
{
    float x;
    float y;
    float z;
} Point;

typedef struct Matrix
{
    Point u;
    Point v;
    Point w;
} Matrix;

typedef struct Ray
{
    Point location;
    Point direction;
} Ray;

typedef struct Pentagon
{
    size_t a;
    size_t b;
    size_t c;
    size_t d;
    size_t e;
} Pentagon;

typedef struct Scene
{
    Point camera;
    float macro;
    float rotX;
    float rotY;

    size_t width;
    size_t height;
    Point points[20];
    Pentagon faces[12];

    nvjpegImage_t data;
} Scen;

#endif