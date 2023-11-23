#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

float A_inv(size_t n, size_t i, size_t j)
{
    return -(i > j ? j * (n - (i - 1.0F)) : i * (n - (j - 1.0F))) / (n + 1.0F);
}

float f(float* y, size_t i)
{
    return expf(-y[i]);
}

float F(float* y, size_t i, float h, size_t n)
{
    return h * h / 12.0F * (f(y, i + 1) + f(y, i - 1) + 10.0F * f(y, i))
        - (i == 1 ? y[0] : i == n - 2 ? y[n - 1] : 0.0F);
}

void diff_solver(float* y, size_t n, float xs, float xe, float a, float b, size_t max_iter)
{
    size_t N = n - 2;
    y[0] = a;
    y[n - 1] = b;

    float h = (xe - xs) / (n - 1);

    float* arr = (float*)malloc(sizeof(float) * N);

    for (size_t iter = 0; iter < max_iter; iter++)
    {
        #pragma omp parallel for
        for (size_t i = 1; i <= N; i++)
        {
            arr[i - 1] = 0.0F;
            for (size_t j = 1; j <= N; j++)
            {
                arr[i - 1] += A_inv(N, i, j) * F(y, j, h, n);
            }
        }
        memcpy(y + 1, arr, sizeof(float) * N);
    }

    free(arr);
}

double measure_time(size_t n, float xs, float xe, float a, float b, size_t max_iter)
{
    float* data = (float*)malloc(sizeof(float) * n);

    double time = omp_get_wtime();
    diff_solver(data, n, xs, xe, a, b, max_iter);
    time = omp_get_wtime() - time;

    free(data);

    return time;
}