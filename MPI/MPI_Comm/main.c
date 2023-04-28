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

float get_sum(int begin, int end)
{
    float sum = 0;
    for (int i = begin; i <= end; i++)
    {
        sum += 1.0F / (float)i;
    }
    return sum;
}

float calc_sum(comm_info_t mpi, int N)
{
    int range = N / mpi.size;
    int begin = mpi.rank * range + 1;
    int end = (mpi.rank == mpi.size - 1) ? N : (mpi.rank + 1) * range;
    float sum = get_sum(begin, end);
    float result = 0.0F;

    MPI_Reduce(&sum, &result, 1, MPI_FLOAT, MPI_SUM, 0, mpi.comm);
    return result;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Input required\n");
        return 1;
    }

    comm_info_t mpi = init_mpi(argc, argv);

    MPI_Comm new_comm;
    int color = (mpi.rank == 0) ? 0 : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi.rank, &new_comm);
    comm_info_t new_mpi = init_comm_mpi(new_comm);

    printf("New comm: rank %d, commsize %d\n", new_mpi.rank, new_mpi.size);

    int N = 0;
    sscanf(argv[1], "%d", &N);

    float sum = calc_sum(new_mpi, N);
    if (mpi.rank == 0) printf("Sum: %.4f\n", sum);

    MPI_Finalize();
    return 0;
}