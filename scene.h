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
    size_t verts[5];
    Point norm;
} Pentagon;

typedef struct Scene
{
    Point camera;
    float macro;
    float rotX;
    float rotY;
    float band;

    float reflectivity;
    size_t reflections;
    size_t width;
    size_t height;
    Point points[20];
    Pentagon faces[12];

    nvjpegImage_t data;
} Scen;

#endif