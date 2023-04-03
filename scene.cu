#ifndef __SCENE_CU__
#define __SCENE_CU__
#include <math.h>
#include "scene.h"

__host__ __device__ float dot(Point *a, Point *b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

__host__ __device__ void normalize(Point *a)
{
    float len = sqrt(dot(a, a));
    a->x /= len;
    a->y /= len;
    a->z /= len;
}

__device__ void scale(Point *dst, float scalar)
{
    dst->x *= scalar;
    dst->y *= scalar;
    dst->z *= scalar;
}

__host__ __device__ void sub(Point *dst, Point *a, Point *b)
{
    dst->x = a->x - b->x;
    dst->y = a->y - b->y;
    dst->z = a->z - b->z;
}

__host__ __device__ void cross(Point *dst, Point *a, Point *b)
{
    dst->x = a->y * b->z - a->z * b->y;
    dst->y = a->z * b->x - a->x * b->z;
    dst->z = a->x * b->y - a->y * b->x;
}

__host__ __device__ void linmap(Point *dst, Matrix *a, Point *x)
{
    dst->x = a->u.x * x->x + a->v.x * x->y + a->w.x * x->z;
    dst->y = a->u.y * x->x + a->v.y * x->y + a->w.y * x->z;
    dst->z = a->u.z * x->x + a->v.z * x->y + a->w.z * x->z;
}

__host__ __device__ void matmul(Matrix *dst, Matrix *a, Matrix *b)
{
    linmap(&dst->u, a, &b->u);
    linmap(&dst->v, a, &b->v);
    linmap(&dst->w, a, &b->w);
}

__host__ __device__ void rotateXY(Matrix *dst, float x, float y)
{
    float sinx = sin(x);
    float cosx = cos(x);
    Matrix leftwards = {
        u : {cosx, 0, -sinx},
        v : {0, 1, 0},
        w : {sinx, 0, cosx},
    };
    float siny = sin(y);
    float cosy = cos(y);
    Matrix upwards = {
        u : {1, 0, 0},
        v : {0, cosy, siny},
        w : {0, -siny, cosy},
    };
    matmul(dst, &leftwards, &upwards);
}

__device__ bool collides(Point *dst, Point *a, Point *b, Point *c, Ray *ray)
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
    dst->x = ray->location.x + ray->direction.x * t;
    dst->y = ray->location.y + ray->direction.y * t;
    dst->z = ray->location.z + ray->direction.z * t;
    return (
        det >= 1e-6 &&
        t > 0.0 &&
        u >= 0.0 && v >= 0.0 && (u + v) <= 1.0);
}

void initializeScene(Scene *img, size_t width, size_t height)
{

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

    img->faces[0] = {verts : {17, 16, 1, 9, 3}};
    img->faces[1] = {verts : {1, 16, 0, 12, 13}};
    img->faces[2] = {verts : {1, 13, 5, 11, 9}};
    img->faces[3] = {verts : {9, 11, 7, 15, 3}};
    img->faces[4] = {verts : {17, 3, 15, 14, 2}};
    img->faces[5] = {verts : {2, 8, 0, 16, 17}};
    img->faces[6] = {verts : {14, 6, 10, 8, 2}};
    img->faces[7] = {verts : {4, 12, 0, 8, 10}};
    img->faces[8] = {verts : {13, 12, 4, 18, 5}};
    img->faces[9] = {verts : {5, 18, 19, 7, 11}};
    img->faces[10] = {verts : {6, 14, 15, 7, 19}};
    img->faces[11] = {verts : {10, 6, 19, 18, 4}};

    // rotate all points based on scene rotation
    Matrix rotation;
    Point tmp;
    rotateXY(&rotation, img->rotX, img->rotY);
    for (size_t i = 0; i < 20; i++)
    {
        linmap(&tmp, &rotation, &img->points[i]);
        img->points[i] = tmp;
    }

    // compute surface normals
    for (size_t i = 0; i < 12; i++)
    {
        Pentagon *face = &img->faces[i];
        Point e1, e2;
        sub(&e1, &img->points[face->verts[1]], &img->points[face->verts[0]]);
        sub(&e2, &img->points[face->verts[3]], &img->points[face->verts[0]]);
        cross(&face->norm, &e1, &e2);
        normalize(&face->norm);
    }
}

__device__ bool castRay(Scene *img, size_t idx, Ray *ray, size_t dull)
{
    for (size_t i = 0; i < 12; i++)
    {
        // check if we collide with a face
        Pentagon face = img->faces[i];
        size_t *verts = &face.verts[0];
        Point collidePoint, collidePointRelative, edge;
        if (
            !collides(&collidePoint, &img->points[verts[0]], &img->points[verts[1]], &img->points[verts[2]], ray) &&
            !collides(&collidePoint, &img->points[verts[0]], &img->points[verts[2]], &img->points[verts[3]], ray) &&
            !collides(&collidePoint, &img->points[verts[0]], &img->points[verts[3]], &img->points[verts[4]], ray))
        {
            continue;
        }

        // check if it hit near an edge
        for (size_t i = 0; i < 5; i++)
        {
            sub(&collidePointRelative, &collidePoint, &img->points[verts[i]]);
            sub(&edge, &img->points[verts[(i + 1) % 5]], &img->points[verts[i]]);
            normalize(&edge);
            scale(&edge, dot(&edge, &collidePointRelative));
            sub(&collidePointRelative, &collidePointRelative, &edge);
            if (dot(&collidePointRelative, &collidePointRelative) < img->band)
            {
                img->data.channel[0][idx] = 255 - dull;
                img->data.channel[1][idx] = 255 - dull;
                img->data.channel[2][idx] = 255 - dull;
                return false;
            }
        }
        // update ray to reflect off surface
        ray->location = collidePoint;
        Point reflected = face.norm;
        scale(&reflected, 2 * dot(&reflected, &ray->direction));
        sub(&ray->direction, &ray->direction, &reflected);
        normalize(&ray->direction);
        return true;
    }
    return false;
}

__device__ void assignColor(Scene *img, size_t idx, float x, float y)
{
    Ray ray = {
        location : img->camera,
    };

    // find initial raycast direction based on x and y coordinate, macro, and camera position
    Matrix rotation;
    rotateXY(&rotation, x * img->macro, y * img->macro);
    Point direction = {0, 0, -1};
    linmap(&ray.direction, &rotation, &direction);

    img->data.channel[0][idx] = 0;
    img->data.channel[1][idx] = 0;
    img->data.channel[2][idx] = 0;
    for (size_t i = 0; i < img->reflections; i++)
    {
        if (!castRay(img, idx, &ray, 25 * i))
        {
            break;
        }
    }
}

#endif