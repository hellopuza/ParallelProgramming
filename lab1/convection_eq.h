#ifndef CONVECTION_EQ
#define CONVECTION_EQ

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

typedef struct
{
    int size;
    int rank;
    MPI_Comm comm;
} comm_info_t;

typedef float (*funcx_t) (float);
typedef float (*funcxy_t) (float, float);

float* solve(funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    float* u = (float*)malloc(sizeof(float) * x_points_num * t_points_num);
    float h = max_x / (float)(x_points_num - 1);
    float tau = max_t / (float)(t_points_num - 1);
    float last_n = t_points_num - 1;
    float last_i = x_points_num - 1;

    float* at_u(int n, int i)
    {
        return &u[n * x_points_num + i];
    }
    float at_t(int n)
    {
        return tau * (float)n;
    }
    float at_x(int i)
    {
        return h * (float)i;
    }
    float at_f(int n, int i)
    {
        return f(at_t(n), at_x(i));
    }

    for (int i = 0; i < x_points_num; i++)
    {
        *at_u(0, i) = phi(at_x(i));
    }
    for (int n = 0; n < t_points_num; n++)
    {
        *at_u(n, 0) = psi(at_t(n));
    }

    for (int i = 1; i < last_i; i++)
    {
        *at_u(1, i) = at_f(0, i) * tau - 0.5F * tau / h * (*at_u(0, i + 1) - *at_u(0, i - 1)) + *at_u(0, i);
    }
    *at_u(1, last_i) = at_f(0, last_i) * tau - tau / h * (*at_u(0, last_i) - *at_u(0, last_i - 1)) + *at_u(0, last_i);

    for (int n = 1; n < last_n; n++)
    {
        *at_u(n, 1) = at_f(n, 0) * h - 0.5F * h / tau * (*at_u(n + 1, 0) - *at_u(n - 1, 0)) + *at_u(n, 0);
    }
    *at_u(last_n, 1) = at_f(last_n, 0) * h - h / tau * (*at_u(last_n, 0) - *at_u(last_n - 1, 0)) + *at_u(last_n, 0);

    for (int n = 2; n <= last_n; n++)
    {
        for (int i = 2; i < last_i; i++)
        {
            *at_u(n, i) = at_f(n - 1, i) * tau * 2.0F - tau / h * (*at_u(n - 1, i + 1) - *at_u(n - 1, i - 1)) + *at_u(n - 2, i);
        }
        *at_u(n, last_i) = at_f(n - 1, last_i) * tau * 2.0F - 2.0F * tau / h * (*at_u(n - 1, last_i) - *at_u(n - 1, last_i - 1)) + *at_u(n - 2, last_i);
    }

    return u;
}

float* solve_parallel(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    float* u = (float*)malloc(sizeof(float) * x_points_num * t_points_num);
    float h = max_x / (float)(x_points_num - 1);
    float tau = max_t / (float)(t_points_num - 1);
    float last_n = t_points_num - 1;
    float last_i = x_points_num - 1;

    float* at_u(int n, int i)
    {
        return &u[n * x_points_num + i];
    }
    float at_t(int n)
    {
        return tau * (float)n;
    }
    float at_x(int i)
    {
        return h * (float)i;
    }
    float at_f(int n, int i)
    {
        return f(at_t(n), at_x(i));
    }

    for (int i = 0; i < x_points_num; i++)
    {
        *at_u(0, i) = phi(at_x(i));
    }
    for (int n = 0; n < t_points_num; n++)
    {
        *at_u(n, 0) = psi(at_t(n));
    }

    for (int i = 1; i < last_i; i++)
    {
        *at_u(1, i) = at_f(0, i) * tau - 0.5F * tau / h * (*at_u(0, i + 1) - *at_u(0, i - 1)) + *at_u(0, i);
    }
    *at_u(1, last_i) = at_f(0, last_i) * tau - tau / h * (*at_u(0, last_i) - *at_u(0, last_i - 1)) + *at_u(0, last_i);

    for (int n = 1; n < last_n; n++)
    {
        *at_u(n, 1) = at_f(n, 0) * h - 0.5F * h / tau * (*at_u(n + 1, 0) - *at_u(n - 1, 0)) + *at_u(n, 0);
    }
    *at_u(last_n, 1) = at_f(last_n, 0) * h - h / tau * (*at_u(last_n, 0) - *at_u(last_n - 1, 0)) + *at_u(last_n, 0);

    for (int n = 2; n <= last_n; n++)
    {
        for (int i = 2; i < last_i; i++)
        {
            *at_u(n, i) = at_f(n - 1, i) * tau * 2.0F - tau / h * (*at_u(n - 1, i + 1) - *at_u(n - 1, i - 1)) + *at_u(n - 2, i);
        }
        *at_u(n, last_i) = at_f(n - 1, last_i) * tau * 2.0F - 2.0F * tau / h * (*at_u(n - 1, last_i) - *at_u(n - 1, last_i - 1)) + *at_u(n - 2, last_i);
    }

    return u;
}

float* get_solution(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    float* data = NULL;
    if (mpi.size == 1)
    {
        data = solve(f, phi, psi, x_points_num, t_points_num, max_x, max_t);
    }
    else
    {
        data = solve_parallel(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t);
    }
    return data;
}

#endif // CONVECTION_EQ