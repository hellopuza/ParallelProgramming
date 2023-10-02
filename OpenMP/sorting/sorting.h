#include <stdlib.h>
#include <string.h>
#include <omp.h>

int min(int a, int b)
{
    return a < b ? a : b;
}

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

void merge_sort_parallel(int* array, int size)
{
    int nthreads = omp_get_num_threads();


}

int check_sorting(int* array, int size)
{
    for (int i = 1; i < size; i++)
    {
        if (array[i - 1] > array[i])
        {
            return 1;
        }
    }
    return 0;
}

