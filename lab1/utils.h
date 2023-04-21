#ifndef UTILS_H
#define UTILS_H

#include "convection_eq.h"

comm_info_t init_comm_mpi(MPI_Comm comm)
{
    comm_info_t mpi;
    MPI_Comm_size(comm, &mpi.size);
    MPI_Comm_rank(comm, &mpi.rank);
    mpi.comm = comm;
    return mpi;
}

comm_info_t init_mpi(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    return init_comm_mpi(MPI_COMM_WORLD);
}

void test_calculation_time(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t, int iterations)
{
    double time = 0.0;

    for (int i = 0; i < iterations; i++)
    {
        MPI_Barrier(mpi.comm);
        double time0 = MPI_Wtime();

        float* data = get_solution(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t);

        double time1 = MPI_Wtime();
        MPI_Barrier(mpi.comm);

        free(data);
        time += time1 - time0;
    }

    if (mpi.rank == 0)
    {
        printf("%.6lf\n", time / iterations);
    }
}

void print_solution(comm_info_t mpi, funcxy_t f, funcx_t phi, funcx_t psi, int x_points_num, int t_points_num, float max_x, float max_t)
{
    float* data = get_solution(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t);

    int range = x_points_num / mpi.size;
    float* line = (float*)malloc(sizeof(float) * x_points_num * mpi.size);

    for (int n = 0; n < t_points_num; n++)
    {
        MPI_Gather(&data[n * x_points_num], x_points_num, MPI_FLOAT, line, x_points_num, MPI_FLOAT, 0, mpi.comm);
        if (mpi.rank == 0)
        {
            for (int i = 0; i < x_points_num; i++)
            {
                int rank = i / range;
                rank = rank > (mpi.size - 1) ? mpi.size - 1 : rank;
                printf("%.6f ", line[rank * x_points_num + i]);
            }
        }

        if (mpi.rank == 0)
        {
            printf("\n");
        }
    }

    free(line);
    free(data);
}

#endif // UTILS_H