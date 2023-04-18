#ifndef SORT_H
#define SORT_H

#include "utils.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void swap(int** a, int** b)
{
    int* t = *a;
    *a = *b;
    *b = t;
}

void merge_sort(int* array, int size)
{
    int* arr_2 = (int*)malloc(sizeof(int) * size);
    int* a = array;
    int* b = arr_2;

    for (int i = 1; i < size; i *= 2)
    {
        for (int j = 0; j < size; j += i * 2)
        {
            int r = j + i;
            int n1 = MIN(i, size - j);
            int n2 = (size < r) ? 0 : MIN(i, size - r);

            for (int ia = 0, ib = 0, k = 0; k < n1 + n2; k++)
            {
                b[j + k] = (ia >= n1) ?              a[r + ib++] :
                           (ib >= n2) ?              a[j + ia++] :
                           (a[j + ia] < a[r + ib]) ? a[j + ia++] :
                                                     a[r + ib++];
            }
        }
        swap(&a, &b);
    }
    swap(&a, &b);

    if (b != array)
    {
        memcpy(array, arr_2, size * sizeof(int));
    }

    free(arr_2);
}

typedef struct
{
    int a;
    int b;
} pair_t;

typedef struct
{
    pair_t* array;
    int size;
} comp_t;

void push(comp_t* comparators, pair_t comp_pair)
{
    comparators->size++;
    comparators->array = realloc(comparators->array, sizeof(pair_t) * comparators->size);
    comparators->array[comparators->size - 1] = comp_pair;
}

void construction(comp_t* comparators, int up_nproc, int* up_range, int down_nproc, int* down_range)
{
    if ((up_nproc == 0) || (down_nproc == 0))
    {
        return;
    }
    if ((up_nproc == 1) && (down_nproc == 1))
    {
        push(comparators, (pair_t){up_range[0], down_range[0]});
        return;
    }

    int up_odd_nproc = up_nproc / 2;
    int down_odd_nproc = down_nproc / 2;
    int up_even_nproc = up_odd_nproc + (up_nproc % 2);
    int down_even_nproc = down_odd_nproc + (down_nproc % 2);

    int* up_even_range = (int*)malloc(sizeof(int) * up_even_nproc);
    int* down_even_range = (int*)malloc(sizeof(int) * down_even_nproc);
    int* up_odd_range = (int*)malloc(sizeof(int) * up_odd_nproc);
    int* down_odd_range = (int*)malloc(sizeof(int) * down_odd_nproc);

    for (int i = 0; i < up_nproc; i++)
    {
        if ((i % 2 == 0) && (up_even_nproc > 0))
        {
            up_even_range[i / 2] = up_range[i];
        }
        else if ((i % 2 == 1) && (up_odd_nproc > 0))
        {
            up_odd_range[i / 2] = up_range[i];
        }
    }
    for (int i = 0; i < down_nproc; i++)
    {
        if ((i % 2 == 0) && (down_even_nproc > 0))
        {
            down_even_range[i / 2] = down_range[i];
        }
        else if ((i % 2 == 1) && (down_odd_nproc > 0))
        {
            down_odd_range[i / 2] = down_range[i];
        }
    }

    construction(comparators, up_even_nproc, up_even_range, down_even_nproc, down_even_range);
    construction(comparators, up_odd_nproc, up_odd_range, down_odd_nproc, down_odd_range);

    for (int i = 1; i < up_nproc + down_nproc - 1; i += 2)
    {
        int j = i + 1;
        pair_t comp_pair;
        comp_pair.a = (i < up_nproc) ? up_range[i] : down_range[i - up_nproc];
        comp_pair.b = (j < up_nproc) ? up_range[j] : down_range[j - up_nproc];
        push(comparators, comp_pair);
    }

    free(up_even_range);
    free(down_even_range);
    free(up_odd_range);
    free(down_odd_range);
}

void partition(comp_t* comparators, int nproc, int* proc_range)
{
    if (nproc == 1)
    {
        return;
    }

    int up_nproc = nproc / 2;
    int down_nproc = up_nproc + (nproc % 2);

    int* up_range = (int*)malloc(sizeof(int) * up_nproc);
    int* down_range = (int*)malloc(sizeof(int) * down_nproc);

    for (int i = 0; i < up_nproc; i++)
    {
        up_range[i] = proc_range[i];
    }
    for (int i = 0; i < down_nproc; i++)
    {
        down_range[i] = proc_range[i + up_nproc];
    }

    partition(comparators, up_nproc, up_range);
    partition(comparators, down_nproc, down_range);
    construction(comparators, up_nproc, up_range, down_nproc, down_range);

    free(up_range);
    free(down_range);
}

void get_comparators(comp_t* comparators, int nproc)
{
    int* proc_range = (int*)malloc(sizeof(int) * nproc);
    for (int i = 0; i < nproc; i++)
    {
        proc_range[i] = i;
    }

    comparators->size = 0;
    comparators->array = NULL;
    partition(comparators, nproc, proc_range);
    free(proc_range);
}

void merge_sort_parallel(int* array, int size)
{
    comm_info_t mpi = init_comm_mpi(MPI_COMM_WORLD);

    uint32_t loc_size = size / mpi.size;

    int* loc_array = (int*)malloc(sizeof(int) * loc_size);
    int* recv_array = (int*)malloc(sizeof(int) * loc_size);
    int* temp_array = (int*)malloc(sizeof(int) * loc_size);
    MPI_Scatter(array, loc_size, MPI_INT, loc_array, loc_size, MPI_INT, 0, mpi.comm);

    merge_sort(loc_array, loc_size);

    comp_t comparators;
    get_comparators(&comparators, mpi.size);

    for (int ncomp = 0; ncomp < comparators.size; ncomp++)
    {
        pair_t comp_pair = comparators.array[ncomp];
        if (mpi.rank == comp_pair.a)
        {
            MPI_Send(loc_array, loc_size, MPI_INT, comp_pair.b, 0, mpi.comm);
            MPI_Recv(recv_array, loc_size, MPI_INT, comp_pair.b, 0, mpi.comm, MPI_STATUS_IGNORE);

            for (int it = 0, il = 0, ir = 0; it < loc_size; it++)
            {
                temp_array[it] = (loc_array[il] < recv_array[ir]) ? loc_array[il++] : recv_array[ir++];
            }
            swap(&loc_array, &temp_array);
        }
        else if (mpi.rank == comp_pair.b)
        {
            MPI_Recv(recv_array, loc_size, MPI_INT, comp_pair.a, 0, mpi.comm, MPI_STATUS_IGNORE);
            MPI_Send(loc_array, loc_size, MPI_INT, comp_pair.a, 0, mpi.comm);

            int start = loc_size - 1;
            for (int it = start, il = start, ir = start; it >= 0; it--)
            {
                temp_array[it] = (loc_array[il] > recv_array[ir]) ? loc_array[il--] : recv_array[ir--];
            }
            swap(&loc_array, &temp_array);
        }
    }

    free(comparators.array);

    MPI_Gather(loc_array, loc_size, MPI_INT, array, loc_size, MPI_INT, 0, mpi.comm);

    free(loc_array);
    free(recv_array);
    free(temp_array);
}

#endif // SORT_H