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

void fft(const float* data, complex_t* out, size_t size)
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

void fft_parallel(const float* data, complex_t* out, size_t size)
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

void fft_parallel_opt(const float* data, complex_t* out, size_t size)
{
    size_t log_n = 0;
    while ((1 << log_n) < size) log_n++;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        out[reverse(i, log_n)] = data[i];
    }

    const size_t end_j = size >> 1;
    for (size_t m = 2; m <= size; m <<= 1)
    {
        const float a = 2.0F * PI / m;
        const size_t n = m >> 1;
        #pragma omp parallel for
        for (size_t j = 0; j < end_j; j += 8)
        {
            __m256i jj = _mm256_set_epi32(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j + 0);
            __m256i jn = _mm256_set_epi32(
                (j + 7) % n, (j + 6) % n, (j + 5) % n, (j + 4) % n,
                (j + 3) % n, (j + 2) % n, (j + 1) % n, (j + 0) % n
            );
            __m256i i1 = _mm256_sub_epi32(_mm256_slli_epi32(jj, 1), jn);
            __m256i i2 = _mm256_add_epi32(i1, _mm256_set1_epi32(n));

            #undef I
            #define R(n,ii) ((float*)&(out[_mm256_extract_epi32((ii),(n))]))[0]
            #define I(n,ii) ((float*)&(out[_mm256_extract_epi32((ii),(n))]))[1]

            __m256 re1 = _mm256_set_ps(R(7,i1), R(6,i1), R(5,i1), R(4,i1), R(3,i1), R(2,i1), R(1,i1), R(0,i1));
            __m256 re2 = _mm256_set_ps(R(7,i2), R(6,i2), R(5,i2), R(4,i2), R(3,i2), R(2,i2), R(1,i2), R(0,i2));
            __m256 im1 = _mm256_set_ps(I(7,i1), I(6,i1), I(5,i1), I(4,i1), I(3,i1), I(2,i1), I(1,i1), I(0,i1));
            __m256 im2 = _mm256_set_ps(I(7,i2), I(6,i2), I(5,i2), I(4,i2), I(3,i2), I(2,i2), I(1,i2), I(0,i2));

            __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(jn), _mm256_set1_ps(a));
            __m256 c = cos256_ps(w);
            __m256 s = sin256_ps(w);

            __m256 t = _mm256_sub_ps(_mm256_mul_ps(re2, c), _mm256_mul_ps(im2, s));
            im2 = _mm256_add_ps(_mm256_mul_ps(re2, s), _mm256_mul_ps(im2, c));
            re2 = t;

            __m256 x11 = _mm256_add_ps(re1, re2);
            __m256 x12 = _mm256_add_ps(im1, im2);
            __m256 x21 = _mm256_sub_ps(re1, re2);
            __m256 x22 = _mm256_sub_ps(im1, im2);

            #define F(n,xii) ((float*)&xii)[n]
            #define C(n) R(n,i1) = F(n,x11); I(n,i1) = F(n,x12); R(n,i2) = F(n,x21); I(n,i2) = F(n,x22);

            C(0); C(1); C(2); C(3); C(4); C(5); C(6); C(7);
        }
    }
}

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