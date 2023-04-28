#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>

typedef double (*func_t) (double);

double integrate(int nproc, func_t f, double a, double b, double err)
{

}

void print_solution(int nproc, func_t f, double a, double b, double err)
{
    double time0 = MPI_Wtime();
    float res = integrate(nproc, f, a, b, err);
    double time1 = MPI_Wtime();

    printf("%lf %lf\n", res, time1 - time0);
}

#endif // INTEGRATION_H