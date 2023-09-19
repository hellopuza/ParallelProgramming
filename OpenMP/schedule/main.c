#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef NUM_ITERS
#define NUM_ITERS 65
#endif

void linear_worker(int t)
{
    for (int i = 0; i < t; i++)
        ;
}

#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#define TEST_SCHED(sched) \
{ \
    int threads[NUM_ITERS]; \
    double time = omp_get_wtime(); \
    DO_PRAGMA(omp parallel for num_threads(NUM_THREADS) sched) \
    for (int i = 0; i < NUM_ITERS; i++) \
    { \
        threads[i] = omp_get_thread_num(); \
        linear_worker(i * 100); \
    } \
    time = omp_get_wtime() - time; \
    printf("%-20s : ", #sched); \
    for (int i = 0; i < NUM_ITERS; i++) \
    { \
        printf("%d ", threads[i]); \
    } \
    printf(" : time %.3e\n", time); \
}

int main()
{
    TEST_SCHED( );
    TEST_SCHED(schedule(static));
    TEST_SCHED(schedule(static, 1));
    TEST_SCHED(schedule(static, 4));
    TEST_SCHED(schedule(dynamic));
    TEST_SCHED(schedule(dynamic, 1));
    TEST_SCHED(schedule(dynamic, 4));
    TEST_SCHED(schedule(guided));
    TEST_SCHED(schedule(guided, 1));
    TEST_SCHED(schedule(guided, 4));
    TEST_SCHED(schedule(runtime));
    TEST_SCHED(schedule(auto));
}

