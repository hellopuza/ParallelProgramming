#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef NUM_ITERS
#define NUM_ITERS 10000
#endif

void linear_worker(int t)
{
    for (int i = 0; i < t; i++) {}
}

#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#define TEST_SCHED(sched) \
{ \
    double time = omp_get_wtime(); \
    DO_PRAGMA(omp parallel for num_threads(NUM_THREADS) sched) \
    for (int i = 0; i < NUM_ITERS; i++) \
        linear_worker(i); \
    printf("%s : %lf\n", #sched, omp_get_wtime() - time); \
}

int main()
{
    TEST_SCHED( );
    TEST_SCHED(schedule(static));
    TEST_SCHED(schedule(static, 1));
    TEST_SCHED(schedule(static, 2));
    TEST_SCHED(schedule(dynamic));
    TEST_SCHED(schedule(dynamic, 1));
    TEST_SCHED(schedule(dynamic, 2));
    TEST_SCHED(schedule(guided));
    TEST_SCHED(schedule(guided, 1));
    TEST_SCHED(schedule(guided, 2));
    TEST_SCHED(schedule(runtime));
    TEST_SCHED(schedule(auto));
}

