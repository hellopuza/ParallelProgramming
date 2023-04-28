#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <mpi.h>

typedef struct
{
    int size;
    int rank;
    MPI_Comm comm;
} comm_info_t;

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

double get_time(comm_info_t mpi)
{
    double time = 0.0;

    int a = 10;

    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        double time0 = MPI_Wtime();
        MPI_Send(&a, 1, MPI_INT, 1, 0, mpi.comm);
        MPI_Recv(&a, 1, MPI_INT, 1, 0, mpi.comm, MPI_STATUS_IGNORE);
        double time1 = MPI_Wtime();
        time = time1 - time0;
    }
    else
    {
        MPI_Recv(&a, 1, MPI_INT, 0, 0, mpi.comm, MPI_STATUS_IGNORE);
        MPI_Send(&a, 1, MPI_INT, 0, 0, mpi.comm);
    }
    MPI_Barrier(mpi.comm);

    return time * 0.5F;
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    double time = 0.0;
    int max_iter = 1000;
    for (int i = 0; i < max_iter; i++)
    {
        time += get_time(mpi);
    }

    if (mpi.rank == 0)
    {
        printf("%lf ms\n", time / max_iter * 1000.0);
    }

    MPI_Finalize();
    return 0;
}