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

__device__ void cross(Point *dst, Point *a, Point *b)
{
    dst->x = a->y * b->z - a->z * b->y;
    dst->y = a->z * b->x - a->x * b->z;
    dst->z = a->x * b->y - a->y * b->x;
}

__device__ void sub(Point *dst, Point *a, Point *b)
{
    dst->x = a->x - b->x;
    dst->y = a->y - b->y;
    dst->z = a->z - b->z;
}

__device__ float dot(Point *a, Point *b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

#endif