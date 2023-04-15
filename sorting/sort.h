#ifndef SORT_H
#define SORT_H

#include "utils.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void merge_sort(int* array, int size)
{
    void swap(int** a, int** b)
    {
        int* t = *a;
        *a = *b;
        *b = t;
    }

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

void merge_sort_parallel(int* array, int size)
{
    void swap(int** a, int** b)
    {
        int* t = *a;
        *a = *b;
        *b = t;
    }

}

#endif // SORT_H