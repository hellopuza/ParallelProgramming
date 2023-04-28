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

float getSum(int begin, int end)
{
    float sum = 0;
    for (int i = begin; i <= end; i++)
    {
        sum += 1.0F / (float)i;
    }
    return sum;
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);
    if (argc < 2)
    {
        printf("Input required\n");
        MPI_Finalize();
        return 1;
    }

    int N = 0;
    sscanf(argv[1], "%d", &N);

    int range = N / mpi.size;
    int begin = mpi.rank * range + 1;
    int end = (mpi.rank == mpi.size - 1) ? N : (mpi.rank + 1) * range;
    float sum = getSum(begin, end);

    if (mpi.rank == 0)
    {
        for (int i = 1; i < mpi.size; i++)
        {
            float increment = 0.0F;
            MPI_Recv(&increment, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += increment;
        }
        printf("%.4f\n", sum);
    }
    else
    {
        MPI_Send(&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}