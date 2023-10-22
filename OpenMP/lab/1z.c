#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ISIZE 1000
#define JSIZE 1000
#define SIZE sizeof(double) * ISIZE * JSIZE

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    double* a = (double*)malloc(SIZE);
    int i, j;
    FILE *ff;
    for (i = 0; i < ISIZE; i++)
    {
        for (j = 0; j < JSIZE; j++)
        {
            a[i * JSIZE + j] = 10 * i + j;
        }
    }

    double time = MPI_Wtime();
#ifndef PARALLEL
    for (i = 2; i < ISIZE; i++)
    {
        for (j = 4; j < JSIZE; j++)
        {
            a[i * JSIZE + j] = sin(4 * a[(i - 2) * JSIZE + j - 4]);
        }
    }
#else
    int size = JSIZE - 4;
    int range = size / mpi_size;
    int begin = mpi_rank * range;
    int end = (mpi_rank == mpi_size - 1) ? size : (mpi_rank + 1) * range;
    for (i = 2; i < ISIZE; i++)
    {
        for (j = begin; j < end; j++)
        {
            a[i * JSIZE + j + 4] = sin(4 * a[(i - 2) * JSIZE + j]);
        }
        for (int r = 0; r < mpi_size; r++)
        {
            if (mpi_rank == r)
            {
                for (int k = 0; k < mpi_size; k++)
                {
                    if (k != r)
                    {
                        int count = (k == mpi_size - 1) ? size - k * range : range;
                        MPI_Recv(&a[i * JSIZE + k * range + 4], count, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            else
            {
                MPI_Send(&a[i * JSIZE + mpi_rank * range + 4], end - begin, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
#endif // PARALLEL
    if (mpi_rank == 0) printf("%lf\n", time - MPI_Wtime());

    if ((argc > 1) && (mpi_rank == 0))
    {
        ff = fopen(argv[1], "wb");
        fwrite(a, sizeof(double), ISIZE * JSIZE, ff);
        fclose(ff);
    }
    free(a);

    MPI_Finalize();
}
