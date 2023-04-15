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
} comm_info_t;

comm_info_t init_mpi(int argc, char* argv[])
{
    comm_info_t mpi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    return mpi;
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

void test_sorting_time(sort_func_t sort_func, gen_seq_func_t gen_seq_func, int initial_size, int end_size, int factor, int iterations)
{
    for (int size = initial_size; size < end_size; size *= factor)
    {
        double time = 0.0;
        for (int i = 0; i < iterations; i++)
        {
            int* array = gen_seq_func(size);

            double time0 = MPI_Wtime();
            sort_func(array, size);
            double time1 = MPI_Wtime();

            free(array);
            time += time1 - time0;
        }
        time /= iterations;

        printf("%d\t%.6lf\n", size, time);
    }
}

#endif // UTILS_H