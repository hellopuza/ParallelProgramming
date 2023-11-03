#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <immintrin.h>
#include "avx_mathfun.h"

#define complex_t complex float

typedef void(*fft_t)(float*, complex_t*, size_t);

void swap(float* a, float* b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

const float PI = acos(-1.0F);

size_t reverse(size_t num, size_t log_n)
{
    size_t res = 0;
    for (size_t i = 0; i < log_n; i++)
    {
        if (num & (1 << i))
        {
            res |= 1 << (log_n - 1 - i);
        }
    }
    return res;
}

void fft(float* data, complex_t* out, size_t size)
{
    size_t log_n = 0;
    while ((1 << log_n) < size) log_n++;

    for (size_t i = 0; i < size; i++)
    {
        out[reverse(i, log_n)] = data[i];
    }

    for (size_t m = 2; m <= size; m <<= 1)
    {
        complex_t a = 2.0F * PI * I / m;
        const size_t n = m >> 1;
        for (size_t k = 0; k < size; k += m)
        {
            for (size_t i = 0; i < n; i++)
            {
                size_t i1 = k + i;
                size_t i2 = i1 + n;
                complex_t x1 = out[i1];
                complex_t x2 = out[i2] * cexpf(a * i);
                out[i1] = x1 + x2;
                out[i2] = x1 - x2;
            }
        }
    }
}

void fft_parallel(float* data, complex_t* out, size_t size)
{
    size_t log_n = 0;
    while ((1 << log_n) < size) log_n++;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        out[reverse(i, log_n)] = data[i];
    }

    for (size_t m = 2; m <= size; m <<= 1)
    {
        complex_t a = 2.0F * PI * I / m;
        const size_t n = m >> 1;
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < size; k += m)
        {
            for (size_t i = 0; i < n; i++)
            {
                size_t i1 = k + i;
                size_t i2 = i1 + n;
                complex_t x1 = out[i1];
                complex_t x2 = out[i2] * cexpf(a * i);
                out[i1] = x1 + x2;
                out[i2] = x1 - x2;
            }
        }
    }
}

void fft_parallel_opt(float* data, complex_t* out, size_t size)
{
    size_t log_n = 0;
    while ((1 << log_n) < size) log_n++;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        out[reverse(i, log_n)] = data[i];
    }

    const size_t end_j = size >> 4;
    for (size_t m = 2; m <= size; m <<= 1)
    {
        const float a = 2.0F * PI / m;
        const size_t n = m >> 1;
        //#pragma omp parallel for
        for (size_t j = 0; j < end_j; j += 8)
        {
            size_t i1 = j + (j / n) * n;
            size_t i2 = i1 + n;

            __attribute__((aligned(32))) float f11[8];
            __attribute__((aligned(32))) float f21[8];
            __attribute__((aligned(32))) float f12[8];
            __attribute__((aligned(32))) float f22[8];

            float* o11 = &out[i1];
            float* o21 = &out[i2];
            float* o12 = &out[i1 + 4];
            float* o22 = &out[i2 + 4];

            size_t count = 8 * sizeof(float);
            memcpy(f11, o11, count);
            memcpy(f21, o21, count);
            memcpy(f12, o12, count);
            memcpy(f22, o22, count);

            __m256 x11 = _mm256_load_ps(f11);
            __m256 x21 = _mm256_load_ps(f21);
            __m256 x12 = _mm256_load_ps(f12);
            __m256 x22 = _mm256_load_ps(f22);

            __m256 re1 = _mm256_shuffle_ps(x11, x12, _MM_SHUFFLE(2, 0, 2, 0));
            __m256 re2 = _mm256_shuffle_ps(x21, x22, _MM_SHUFFLE(2, 0, 2, 0));
            __m256 im1 = _mm256_shuffle_ps(x11, x12, _MM_SHUFFLE(3, 1, 3, 1));
            __m256 im2 = _mm256_shuffle_ps(x21, x22, _MM_SHUFFLE(3, 1, 3, 1));

            __m256 w = _mm256_set_ps(
                (j + 7) % n, (j + 6) % n, (j + 3) % n, (j + 2) % n,
                (j + 5) % n, (j + 4) % n, (j + 1) % n, (j + 0) % n
            );
            w = _mm256_mul_ps(w, _mm256_set1_ps(a));
            __m256 c = cos256_ps(w);
            __m256 s = sin256_ps(w);

            __m256 t = _mm256_sub_ps(_mm256_mul_ps(re2, c), _mm256_mul_ps(im2, s));
            im2 = _mm256_add_ps(_mm256_mul_ps(re2, s), _mm256_mul_ps(im2, c));
            re2 = t;

            x11 = _mm256_add_ps(re1, re2);
            x12 = _mm256_add_ps(im1, im2);
            x21 = _mm256_sub_ps(re1, re2);
            x22 = _mm256_sub_ps(im1, im2);

            o11[0] = ((float*)&x11)[0]; o11[1] = ((float*)&x12)[0];
            o11[2] = ((float*)&x11)[1]; o11[3] = ((float*)&x12)[1];
            o11[4] = ((float*)&x11)[4]; o11[5] = ((float*)&x12)[4];
            o11[6] = ((float*)&x11)[5]; o11[7] = ((float*)&x12)[5];

            o12[0] = ((float*)&x11)[2]; o12[1] = ((float*)&x12)[2];
            o12[2] = ((float*)&x11)[3]; o12[3] = ((float*)&x12)[3];
            o12[4] = ((float*)&x11)[6]; o12[5] = ((float*)&x12)[6];
            o12[6] = ((float*)&x11)[7]; o12[7] = ((float*)&x12)[7];

            o21[0] = ((float*)&x21)[0]; o21[1] = ((float*)&x22)[0];
            o21[2] = ((float*)&x21)[1]; o21[3] = ((float*)&x22)[1];
            o21[4] = ((float*)&x21)[4]; o21[5] = ((float*)&x22)[4];
            o21[6] = ((float*)&x21)[5]; o21[7] = ((float*)&x22)[5];

            o22[0] = ((float*)&x21)[2]; o22[1] = ((float*)&x22)[2];
            o22[2] = ((float*)&x21)[3]; o22[3] = ((float*)&x22)[3];
            o22[4] = ((float*)&x21)[6]; o22[5] = ((float*)&x22)[6];
            o22[6] = ((float*)&x21)[7]; o22[7] = ((float*)&x22)[7];
        }
    }
}
//0.0 1.0 4.0 5.0 2.0 3.0 6.0 7.0                                                                                         
//0.5 1.5 4.5 5.5 2.5 3.5 6.5 7.5

double measure_time(fft_t func, size_t size)
{
    float* data = (float*)malloc(sizeof(float) * size);
    complex_t* out = (complex_t*)malloc(sizeof(complex_t) * size);

    double time = omp_get_wtime();
    func(data, out, size);
    time = omp_get_wtime() - time;

    free(data);
    free(out);

    return time;
}