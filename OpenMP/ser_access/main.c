#include <omp.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    int var = 0;

    #pragma omp parallel for ordered shared(var)
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        #pragma omp ordered
        {
            var++;
        }

        int current_thread = omp_get_thread_num();
        printf("thread: %d, var: %d\n", current_thread, var);
    }

    return 0;
}
