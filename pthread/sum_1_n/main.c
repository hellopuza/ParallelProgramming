#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef void* (*routine_t)(void*);

typedef struct
{
    int id;
    int num_thr;
    int N;
    double sum;
} thread_data_t;

void* sum(void* arg)
{
    thread_data_t* data = (thread_data_t*)arg;
    int range = data->N / data->num_thr;
    int begin = data->id * range + 1;
    int end = (data->id == data->num_thr - 1) ? data->N : (data->id + 1) * range;

    data->sum = 0.0;
    for (int i = begin; i <= end; i++)
    {
        data->sum += 1.0 / (double)i;
    }
    return NULL;
}

void run_threads(int num_thr, routine_t routine, thread_data_t* arg)
{
    pthread_t* th = (pthread_t*)malloc(num_thr * sizeof(pthread_t));
    for (int i = 0; i < num_thr; i++)
    {
        int rc = 0;
        if (rc = pthread_create(&th[i], NULL, routine, &arg[i]))
        {
            fprintf(stderr, "Error: pthread_create %d\n", rc);
            exit(1);
        }
    }

    for (int i = 0; i < num_thr; i++)
    {
        pthread_join(th[i], NULL);
    }
    free(th);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Input required\n");
        return 1;
    }
    int num_thr = 0;
    int N = 0;
    sscanf(argv[1], "%d", &num_thr);
    sscanf(argv[2], "%d", &N);

    thread_data_t* arg = (thread_data_t*)malloc(num_thr * sizeof(thread_data_t));
    for (int i = 0; i < num_thr; i++)
    {
        arg[i].id = i;
        arg[i].num_thr = num_thr;
        arg[i].N = N;
    }

    run_threads(num_thr, sum, arg);

    for (int i = 1; i < num_thr; i++)
    {
        arg[0].sum += arg[i].sum;
    }
    printf("%lf\n", arg[0].sum);

    free(arg);
}