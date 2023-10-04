#include "sorting.h"
#include <stdio.h>
#include <time.h>

double test_sorting(size_t size, sort_t func)
{
    srand(0);
    int* arr = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand();
    }

    double time = omp_get_wtime();
    func(arr, size);
    time = omp_get_wtime() - time;

#ifdef DEBUG
    if (check_sorting(arr, size))
    {
        printf("Wrong sorting\n");
        exit(1);
    }
#endif

    free(arr);
    return time;
}

#define PRINT_SORT_TEST(func) \
    printf("%-30s : %lf\n", #func, test_sorting(size, func));

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Array size and threads num required\n");
        exit(1);
    }

    int size = 0;
    int nproc = 0;
    sscanf(argv[1], "%d", &size);
    sscanf(argv[2], "%d", &nproc);

    omp_set_num_threads(nproc);

    PRINT_SORT_TEST(merge_sort);
    PRINT_SORT_TEST(merge_sort_parallel);
    PRINT_SORT_TEST(quick_sort);
    PRINT_SORT_TEST(quick_sort_parallel);
}

