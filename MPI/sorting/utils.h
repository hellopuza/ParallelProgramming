#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

typedef void (*sort_func_t) (int*, int);
typedef int* (*gen_seq_func_t) (int);

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

int* gen_seq_rnd(int size)
{
    srand(time(NULL));

    int* array = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++)
    {
        array[i] = rand();
    }
    return array;
}

int* gen_seq_best(int size)
{
    int* array = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i += 10)
    {
        array[i] = i + (rand() % 10);
    }
    return array;
}

int* gen_seq_worst(int size)
{
    srand(time(NULL));

    int* array = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() * 2;
    }
    return array;
}

int test_seq_sorted(int* array, int size)
{
    for (int i = 1; i < size; i++)
    {
        if (array[i] < array[i - 1])
        {
            return 0;
        }
    }
    return 1;
}

void test_sorting_time(comm_info_t mpi, sort_func_t sort_func, gen_seq_func_t gen_seq_func, int initial_size, int end_size, float factor, int iterations)
{
    for (int size = initial_size; size < end_size; size = (int)((float)size * factor))
    {
        int right_size = (size / mpi.size + (size % mpi.size != 0)) * mpi.size;
        double time = 0.0;

        for (int i = 0; i < iterations; i++)
        {
            int* array = NULL;
            if (mpi.rank == 0)
            {
                array = gen_seq_func(right_size);
            }

            MPI_Barrier(mpi.comm);
            double time0 = MPI_Wtime();

            sort_func(array, right_size);

            double time1 = MPI_Wtime();
            MPI_Barrier(mpi.comm);

            #if (TEST_SORTING_CORRECTNESS)
            if ((mpi.rank == 0) && !test_seq_sorted(array, right_size))
            {
                printf("not sorted\n");
                exit(-1);
            }
            #endif // TEST_SORTING_CORRECTNESS

            free(array);
            time += time1 - time0;
        }

        if (mpi.rank == 0)
        {
            printf("%d\t%.6lf\n", right_size, time / iterations);
        }
    }
}

#endif // UTILS_H