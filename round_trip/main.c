#include <mpi.h>
#include <stdio.h>

typedef struct
{
    int size;
    int rank;
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

    if (mpi.rank == 0)
    {
        int N = 0;
        MPI_Send(&N, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    int N = 0;
    MPI_Recv(&N, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank: %d received: %d\n", mpi.rank, N);

    N += 1;
    if (mpi.rank != 0)
    {
        int next_rank = (mpi.rank + 1) % mpi.size;
        MPI_Send(&N, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}