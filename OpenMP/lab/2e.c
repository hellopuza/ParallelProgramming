#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define ISIZE 1000
#define JSIZE 1000
#define SIZE sizeof(double) * ISIZE * JSIZE

int main(int argc, char **argv)
{
    double* a = (double*)malloc(SIZE);
    int i, j;
    FILE *ff;
    for (i = 0; i < ISIZE; i++)
    {
        for (j = 0; j < JSIZE; j++)
        {
            a[i * JSIZE + j] = 10 * i + j;
        }
    }

    double time = omp_get_wtime();
#ifndef PARALLEL
    for (i = 0; i < ISIZE - 4; i++)
    {
        for (j = 0; j < JSIZE - 2; j++)
        {
            a[i * JSIZE + j] = sin(0.1 * a[(i + 4) * JSIZE + j + 2]);
        }
    }
#else
    double* b = (double*)malloc(SIZE);
    memcpy(b, a, SIZE);
    #pragma omp parallel for collapse(2) num_threads(atoi(argv[1]))
    for (i = 0; i < ISIZE - 4; i++)
    {
        for (j = 0; j < JSIZE - 2; j++)
        {
            a[i * JSIZE + j] = sin(0.1 * b[(i + 4) * JSIZE + j + 2]);
        }
    }
    free(b);
#endif // PARALLEL
    printf("%lf\n", time - omp_get_wtime());

    if (argc > 2)
    {
        ff = fopen(argv[2], "wb");
        fwrite(a, sizeof(double), ISIZE * JSIZE, ff);
        fclose(ff);
    }
    free(a);
}
