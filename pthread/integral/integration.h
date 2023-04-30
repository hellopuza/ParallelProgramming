#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <semaphore.h>

typedef void* (*routine_t)(void*);
typedef double (*func_t)(double);

typedef struct
{
    double a;
    double b;
    double f_a;
    double f_b;
    double s;
} integral_data_t;

typedef struct
{
    integral_data_t* data;
    int sp;
    int max_size;
} integral_stack_t;

typedef struct
{
    int num_thr;
    func_t f;
    int big_stack_size;
    int num_thr_active;

    double s;
    double err;
    integral_stack_t stk;

    pthread_mutex_t lock_stk;
    pthread_mutex_t lock_s;
    sem_t sem_tasks_num;
} thread_data_t;

void run_threads(int num_thr, routine_t routine, thread_data_t* arg)
{
    pthread_t* th = (pthread_t*)malloc(num_thr * sizeof(pthread_t));
    for (int i = 0; i < num_thr; i++)
    {
        int rc = 0;
        if (rc = pthread_create(&th[i], NULL, routine, arg))
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

double trapezoidal(double a, double b, double f_a, double f_b)
{
    return (f_a + f_b) * (b - a) * 0.5;
}

int accuracy_is_good(double old_val, double new_val, double err)
{
    return fabs(1.0 - old_val / new_val) < err;
}

int stack_size(double a, double b)
{
    return 64 + (int)ceil(fabs(log2(b - a)));
}

int big_stack_size(double a, double b, double err)
{
    return (int)ceil(fabs(log2((b - a) / err)));
}

void stack_init(integral_stack_t* stk, int size)
{
    stk->data = (integral_data_t*)malloc(sizeof(integral_data_t) * size);
    stk->sp = 0;
    stk->max_size = size;
}

void stack_destroy(integral_stack_t* stk)
{
    free(stk->data);
}

void stack_push(integral_stack_t* stk, integral_data_t data)
{
    stk->data[stk->sp] = data;
    stk->sp++;
}

void stack_pop(integral_stack_t* stk, integral_data_t* data)
{
    stk->sp--;
    *data = stk->data[stk->sp];
}

int stack_is_free(integral_stack_t* stk)
{
    return stk->sp == 0;
}

int stack_is_full(integral_stack_t* stk)
{
    return stk->sp == stk->max_size;
}

int stack_is_big(thread_data_t* global, integral_stack_t* stk)
{
    return stk->sp == global->big_stack_size;
}

int load_global_stack(thread_data_t* global, integral_data_t* data)
{
    sem_wait(&global->sem_tasks_num);
    pthread_mutex_lock(&global->lock_stk);

    stack_pop(&global->stk, data);
    if (!stack_is_free(&global->stk))
    {
        sem_post(&global->sem_tasks_num);
    }

    if (data->a <= data->b)
    {
        global->num_thr_active++;
    }

    pthread_mutex_unlock(&global->lock_stk);

    return data->a > data->b;
}

void store_local_stack(thread_data_t* global, integral_stack_t* stk)
{
    pthread_mutex_lock(&global->lock_stk);
    int global_stack_was_free = stack_is_free(&global->stk);

    integral_data_t data;
    while ((stk->sp > 1) && !stack_is_full(&global->stk))
    {
        stack_pop(stk, &data);
        stack_push(&global->stk, data);
    }

    if (global_stack_was_free)
    {
        sem_post(&global->sem_tasks_num);
    }

    pthread_mutex_unlock(&global->lock_stk);
}

void write_terminals(thread_data_t* global)
{
    pthread_mutex_lock(&global->lock_stk);

    global->num_thr_active--;
    if (stack_is_free(&global->stk) && (global->num_thr_active == 0))
    {
        for (int i = 0; i < global->num_thr; i++)
        {
            stack_push(&global->stk, (integral_data_t){1.0, 0.0, 0.0, 0.0, 0.0});
        }
        sem_post(&global->sem_tasks_num);
    }

    pthread_mutex_unlock(&global->lock_stk);
}

void* integral_routine(void* arg)
{
    thread_data_t* global = (thread_data_t*)arg;

    integral_stack_t stk;
    stack_init(&stk, global->stk.max_size);
    integral_data_t data = (integral_data_t){0.0, 0.0, 0.0, 0.0, 0.0};
    double local_s = 0.0;

    while (1)
    {
        if (load_global_stack(global, &data))
        {
            break;
        }

        while (1)
        {
            double c = (data.a + data.b) * 0.5;
            double f_c = global->f(c);

            double s_ac = trapezoidal(data.a, c, data.f_a, f_c);
            double s_cb = trapezoidal(c, data.b, f_c, data.f_b);
            double s_new = s_ac + s_cb;

            if (accuracy_is_good(data.s, s_new, global->err))
            {
                local_s += s_new;

                if (stack_is_free(&stk))
                {
                    break;
                }
                stack_pop(&stk, &data);
            }
            else
            {
                stack_push(&stk, (integral_data_t){data.a, c, data.f_a, f_c, s_ac});
                data.a = c;
                data.f_a = f_c;
                data.s = s_cb;
            }

            if (stack_is_big(global, &stk))
            {
                store_local_stack(global, &stk);
            }
        }

        write_terminals(global);
    }

    pthread_mutex_lock(&global->lock_s);
    global->s += local_s;
    pthread_mutex_unlock(&global->lock_s);

    stack_destroy(&stk);
    return NULL;
}

double integrate(int num_thr, func_t f, double a, double b, double err)
{
    thread_data_t arg;
    arg.num_thr = num_thr;
    arg.f = f;
    arg.big_stack_size = big_stack_size(a, b, err);
    arg.num_thr_active = 0;
    arg.s = 0.0;
    arg.err = err;

    stack_init(&arg.stk, stack_size(a, b));
    stack_push(&arg.stk, (integral_data_t){a, b, f(a), f(b), trapezoidal(a, b, f(a), f(b))});

    pthread_mutex_init(&arg.lock_stk, NULL);
    pthread_mutex_init(&arg.lock_s, NULL);
    sem_init(&arg.sem_tasks_num, 0, 1);

    run_threads(num_thr, integral_routine, &arg);

    stack_destroy(&arg.stk);
    pthread_mutex_destroy(&arg.lock_stk);
    pthread_mutex_destroy(&arg.lock_s);
    sem_destroy(&arg.sem_tasks_num);

    return arg.s;
}

#endif // INTEGRATION_H