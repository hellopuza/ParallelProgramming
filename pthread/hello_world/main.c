#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef void* (*routine_t)(void*);

typedef struct
{
    int id;
    int num_thr;
} thread_data_t;

void* hello(void* arg)
{
    thread_data_t* data = (thread_data_t*)arg;
    printf("Hello, world! thread id = %d, threads number = %d\n", data->id, data->num_thr);
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
    if (argc < 2)
    {
        printf("Input required\n");
        return 1;
    }
    int num_thr = 0;
    sscanf(argv[1], "%d", &num_thr);

    thread_data_t* arg = (thread_data_t*)malloc(num_thr * sizeof(thread_data_t));
    for (int i = 0; i < num_thr; i++)
    {
        arg[i].id = i;
        arg[i].num_thr = num_thr;
    }
    run_threads(num_thr, hello, arg);
    free(arg);
}