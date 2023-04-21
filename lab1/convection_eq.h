#ifndef CONVECTION_EQ
#define CONVECTION_EQ

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#define LAMBDAS \
    float* at_u(int n, int i) { return &u[n * x_points_num + i]; } \
    float at_t(int n) { return tau * (float)n; } \
    float at_x(int i) { return h * (float)(i); } \
    float at_f(int n, int i) { return f(at_t(n), at_x(i)); } \
    float scheme_left(int n) { return psi(at_t(n)); } \
    float scheme_bottom(int i) { return phi(at_x(i)); } \
    float scheme_right_1(int n, int i) { return at_f(n - 1, i) * tau - tau / h * (*at_u(n - 1, i) - *at_u(n - 1, i - 1)) + *at_u(n - 1, i); } \
    float scheme_right_2(int n, int i) { return at_f(n - 1, i) * tau * 2.0F - 2.0F * tau / h * (*at_u(n - 1, i) - *at_u(n - 1, i - 1)) + *at_u(n - 2, i); } \
    float scheme_middle_1(int n, int i) { return at_f(n - 1, i) * tau - 0.5F * tau / h * (*at_u(n - 1, i + 1) - *at_u(n - 1, i - 1)) + *at_u(n - 1, i); } \
    float scheme_middle_2(int n, int i) { return at_f(n - 1, i) * tau * 2.0F - tau / h * (*at_u(n - 1, i + 1) - *at_u(n - 1, i - 1)) + *at_u(n - 2, i); }

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

    LAMBDAS

    for (int i = 0; i < x_points_num; i++)
    {
        *at_u(0, i) = scheme_bottom(i);
    }
    for (int n = 0; n < t_points_num; n++)
    {
        *at_u(n, 0) = scheme_left(n);
    }

    for (int i = 1; i < last_i; i++)
    {
        *at_u(1, i) = scheme_middle_1(1, i);
    }
    *at_u(1, last_i) = scheme_right_1(1, last_i);

    for (int n = 2; n <= last_n; n++)
    {
        for (int i = 1; i < last_i; i++)
        {
            *at_u(n, i) = scheme_middle_2(n, i);
        }
        *at_u(n, last_i) = scheme_right_2(n, last_i);
    }

    return u;
}

float* solve_parallel(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    float* u = (float*)malloc(sizeof(float) * x_points_num * t_points_num);
    float h = max_x / (float)(x_points_num - 1);
    float tau = max_t / (float)(t_points_num - 1);
    int last_n = t_points_num - 1;
    int last_i = x_points_num - 1;

    int range = x_points_num / mpi.size;
    int begin_i = mpi.rank * range;
    int end_i = ((mpi.rank == mpi.size - 1) ? x_points_num : (mpi.rank + 1) * range) - 1;
    int loc_num = end_i - begin_i + 1;

    LAMBDAS

    void send_recv_bounds(int n)
    {
        if ((begin_i != 0) && (mpi.rank % 2 == 1)) MPI_Send(at_u(n, begin_i), 1, MPI_FLOAT, mpi.rank - 1, 0, mpi.comm);
        if ((end_i != last_i) && (mpi.rank % 2 == 0)) MPI_Recv(at_u(n, end_i + 1), 1, MPI_FLOAT, mpi.rank + 1, 0, mpi.comm, MPI_STATUS_IGNORE);
        if ((begin_i != 0) && (mpi.rank % 2 == 0)) MPI_Send(at_u(n, begin_i), 1, MPI_FLOAT, mpi.rank - 1, 0, mpi.comm);
        if ((end_i != last_i) && (mpi.rank % 2 == 1)) MPI_Recv(at_u(n, end_i + 1), 1, MPI_FLOAT, mpi.rank + 1, 0, mpi.comm, MPI_STATUS_IGNORE);

        if ((end_i != last_i) && (mpi.rank % 2 == 0)) MPI_Send(at_u(n, end_i), 1, MPI_FLOAT, mpi.rank + 1, 0, mpi.comm);
        if ((begin_i != 0) && (mpi.rank % 2 == 1)) MPI_Recv(at_u(n, begin_i - 1), 1, MPI_FLOAT, mpi.rank - 1, 0, mpi.comm, MPI_STATUS_IGNORE);
        if ((end_i != last_i) && (mpi.rank % 2 == 1)) MPI_Send(at_u(n, end_i), 1, MPI_FLOAT, mpi.rank + 1, 0, mpi.comm);
        if ((begin_i != 0) && (mpi.rank % 2 == 0)) MPI_Recv(at_u(n, begin_i - 1), 1, MPI_FLOAT, mpi.rank - 1, 0, mpi.comm, MPI_STATUS_IGNORE);
    }

    for (int i = begin_i; i <= end_i; i++)
    {
        *at_u(0, i) = scheme_bottom(i);
    }
    send_recv_bounds(0);

    for (int i = begin_i + 1; i <= end_i - 1; i++)
    {
        *at_u(1, i) = scheme_middle_1(1, i);
    }
    *at_u(1, begin_i) = (begin_i == 0) ? scheme_left(1) : scheme_middle_1(1, begin_i);
    *at_u(1, end_i) = (end_i == last_i) ? scheme_right_1(1, last_i) : scheme_middle_1(1, end_i);
    send_recv_bounds(1);

    for (int n = 2; n <= last_n; n++)
    {
        for (int i = begin_i + 1; i <= end_i - 1; i++)
        {
            *at_u(n, i) = scheme_middle_2(n, i);
        }
        *at_u(n, begin_i) = (begin_i == 0) ? scheme_left(n) : scheme_middle_2(n, begin_i);
        *at_u(n, end_i) = (end_i == last_i) ? scheme_right_2(n, last_i) : scheme_middle_2(n, end_i);
        send_recv_bounds(n);
    }

    return u;
}

float* get_solution(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    return (mpi.size == 1) ? solve(f, phi, psi, x_points_num, t_points_num, max_x, max_t) :
                             solve_parallel(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t);
}

#endif // CONVECTION_EQ