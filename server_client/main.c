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

comm_info_t init_mpi(int argc, char* argv[])
{
    comm_info_t mpi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    return mpi;
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    MPI_Comm comm;

    MPI_Comm_spawn("server", MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm, MPI_ERRCODES_IGNORE);
    MPI_Comm_spawn("client", MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm, MPI_ERRCODES_IGNORE);

    MPI_Finalize();
    return 0;
}