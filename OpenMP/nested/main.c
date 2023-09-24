#include <omp.h>
#include <stdio.h>

#ifndef MAX_DEPTH
#define MAX_DEPTH 3
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 3
#endif

void nested_parallel(int depth)
{
    if (depth < 1)
    {
        return;
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        printf("level %d, thread %d / %d, ancestor %d, threads per current level %d\n",
                omp_get_level(),
                omp_get_thread_num(),
                omp_get_num_threads(),
                omp_get_ancestor_thread_num(omp_get_level() - 1),
                omp_get_level() * omp_get_num_threads());

        #pragma omp barrier
        nested_parallel(depth - 1);
    }
}

int main(int argc, char* argv[])
{
    omp_set_nested(1);
    nested_parallel(MAX_DEPTH);

    return 0;
}
