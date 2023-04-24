#ifndef LIFE_H
#define LIFE_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

typedef struct
{
    int size;
    int rank;
    MPI_Comm comm;
} comm_info_t;

void create_grid(int8_t* data[2], int size)
{
    data[0] = (int8_t*)malloc(size * size);
    data[1] = (int8_t*)malloc(size * size);
}

void destroy_grid(int8_t* data[2])
{
    free(data[0]);
    free(data[1]);
}

int8_t* at(int8_t* data, int size, int x, int y)
{
    x = (x < 0) ? size + x : x;
    x = (x >= size) ? x - size : x;
    y = (y < 0) ? size + y : y;
    y = (y >= size) ? y - size : y;
    return &data[size * y + x];
}

void swap_ptr(int8_t** a, int8_t** b)
{
    int8_t* temp = *a;
    *a = *b;
    *b = temp;
}

int8_t sum8(int8_t* data, int size, int x, int y)
{
    int8_t sum = 0;
    for (int ix = -1; ix < 2; ix++)
    {
        for (int iy = -1; iy < 2; iy++)
        {
            sum += *at(data, size, x + ix, y + iy);
        }
    }

    return sum - *at(data, size, x, y);
}

void circular_trip(comm_info_t mpi, void* sendbuf, void* recvbuf, int size, int step)
{
    int next_rank = (mpi.rank + step) % mpi.size;
    next_rank = next_rank < 0 ? mpi.size + next_rank : next_rank;

    int prev_rank = (mpi.rank - step) % mpi.size;
    prev_rank = prev_rank < 0 ? mpi.size + prev_rank : prev_rank;

    if (mpi.rank % 2 == 0) MPI_Send(sendbuf, size, MPI_CHAR, next_rank, 0, mpi.comm);
    if (mpi.rank % 2 == 1) MPI_Recv(recvbuf, size, MPI_CHAR, prev_rank, 0, mpi.comm, MPI_STATUS_IGNORE);
    if (mpi.rank % 2 == 1) MPI_Send(sendbuf, size, MPI_CHAR, next_rank, 0, mpi.comm);
    if (mpi.rank % 2 == 0) MPI_Recv(recvbuf, size, MPI_CHAR, prev_rank, 0, mpi.comm, MPI_STATUS_IGNORE);
}

void life_step_serial(int8_t* data[2], int size)
{
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            int8_t sum = sum8(data[0], size, x, y);
            int8_t center = *at(data[0], size, x, y);

            int8_t lives = center && ((sum == 2) || (sum == 3));
            int8_t born = !center && (sum == 3);
            *at(data[1], size, x, y) = born || lives;
        }
    }
    swap_ptr(&data[0], &data[1]);
}

void life_step_parallel(comm_info_t mpi, int8_t* data[2], int size)
{
    int range = size / mpi.size;
    int begin_y = mpi.rank * range;
    int end_y = ((mpi.rank == mpi.size - 1) ? size : (mpi.rank + 1) * range) - 1;

    circular_trip(mpi, at(data[0], size, 0, end_y), at(data[0], size, 0, begin_y - 1), size, 1);
    circular_trip(mpi, at(data[0], size, 0, begin_y), at(data[0], size, 0, end_y + 1), size, -1);

    for (int y = begin_y; y <= end_y; y++)
    {
        for (int x = 0; x < size; x++)
        {
            int8_t sum = sum8(data[0], size, x, y);
            int8_t center = *at(data[0], size, x, y);

            int8_t lives = center && ((sum == 2) || (sum == 3));
            int8_t born = !center && (sum == 3);
            *at(data[1], size, x, y) = born || lives;
        }
    }
    swap_ptr(&data[0], &data[1]);
}

void life_step(comm_info_t mpi, int8_t* data[2], int size)
{
    if (mpi.size == 1)
    {
        life_step_serial(data, size);
    }
    else
    {
        life_step_parallel(mpi, data, size);
    }
}

void gather_data(comm_info_t mpi, int8_t* data, int size)
{
    int range = size / mpi.size;
    int begin_y = mpi.rank * range;
    int end_y = ((mpi.rank == mpi.size - 1) ? size : (mpi.rank + 1) * range) - 1;
    int loc_size = (end_y - begin_y + 1) * size;
    int offset = begin_y * size;

    int* recvcounts = (mpi.rank == 0) ? (int*)malloc(sizeof(int) * mpi.size) : NULL;
    int* displs = (mpi.rank == 0) ? (int*)malloc(sizeof(int) * mpi.size) : NULL;

    MPI_Gather(&loc_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, mpi.comm);
    MPI_Gather(&offset, 1, MPI_INT, displs, 1, MPI_INT, 0, mpi.comm);
    MPI_Gatherv(at(data, size, 0, begin_y), loc_size, MPI_CHAR, data, recvcounts, displs, MPI_CHAR, 0, mpi.comm);

    free(recvcounts);
    free(displs);
}

#endif // LIFE_H