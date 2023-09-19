#include <omp.h>
#include <stdio.h>

int main()
{
#pragma omp parallel
    {
        int threads_num = omp_get_num_threads();
        int current_thread = omp_get_thread_num();
        printf("Hello World, threads num: %d, current_thread %d\n", threads_num, current_thread);
    }
    return 0;
}
