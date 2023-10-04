#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef void (*sort_t)(int*, int);

int min(int a, int b)
{
    return a < b ? a : b;
}

void swap_p(int** a, int** b)
{
    int* t = *a;
    *a = *b;
    *b = t;
}

void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

void merge_sort(int* array, int size)
{
    int* arr2 = (int*)malloc(sizeof(int) * size);
    int* a = array;
    int* b = arr2;

    for (int i = 1; i < size; i *= 2)
    {
        for (int j = 0; j < size; j += i * 2)
        {
            int r = j + i;
            int n1 = min(i, size - j);
            int n2 = (size < r) ? 0 : min(i, size - r);

            for (int ia = 0, ib = 0, k = 0; k < n1 + n2; k++)
            {
                b[j + k] = (ia >= n1) ?              a[r + ib++] :
                           (ib >= n2) ?              a[j + ia++] :
                           (a[j + ia] < a[r + ib]) ? a[j + ia++] :
                                                     a[r + ib++];
            }
        }
        swap_p(&a, &b);
    }
    swap_p(&a, &b);

    if (b != array)
    {
        memcpy(array, arr2, size * sizeof(int));
    }

    free(arr2);
}

void quick_sort(int* array, int size)
{
    if (size < 2)
    {
        return;
    }
    int pivot = array[0];

    int i = -1;
    int j = size;
    while (1)
    {
        do j--; while (array[j] > pivot);
        do i++; while (array[i] < pivot);

        if (i >= j)
        {
            break;
        }
        swap(&(array[i]), &(array[j]));
    }

    j++;
    quick_sort(array, j);
    quick_sort(array + j, size - j);
}

int MIN_SIZE = 1024;

void merge_sort_task_parallel(int* array, int size, sort_t func)
{
    omp_set_nested(1);
    if (size <= MIN_SIZE)
    {
        func(array, size);
        return;
    }

    int size0 = size / 2;
    int size1 = size0 + size % 2;

    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task
        merge_sort_task_parallel(array, size0, func);
        #pragma omp task
        merge_sort_task_parallel(array + size0, size1, func);
    }

    int* arr2 = (int*)malloc(sizeof(int) * size);
    memcpy(arr2, array, size * sizeof(int));

    for (int k = 0, i = 0, j = size0; k < size; k++)
    {
        array[k] = (i >= size0)        ? arr2[j++] :
                   (j >= size)         ? arr2[i++] :
                   (arr2[i] < arr2[j]) ? arr2[i++] :
                                         arr2[j++];
    }
    free(arr2);
}

void merge_sort_parallel_test(int* array, int size)
{
    merge_sort_task_parallel(array, size, merge_sort);
}

void merge_sort_parallel(int* array, int size)
{
    MIN_SIZE = size / omp_get_max_threads();
    merge_sort_task_parallel(array, size, merge_sort);
}

void quick_sort_parallel_test(int* array, int size)
{
    merge_sort_task_parallel(array, size, quick_sort);
}

void quick_sort_parallel(int* array, int size)
{
    MIN_SIZE = size / omp_get_max_threads();
    merge_sort_task_parallel(array, size, quick_sort);
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

