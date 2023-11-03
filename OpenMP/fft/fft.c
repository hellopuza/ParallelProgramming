#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <smmintrin.h>

#include "avx_mathfun.h"

#define complex_t complex float

union vec8
{
    __m256 m;
    float v[8];
};

typedef void(*fft_t)(float*, complex_t*, size_t);

void swap(float* a, float* b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

const float PI2 = acos(-1.0F) * 2.0F;

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
        complex_t a = PI2 * I / m;
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
        complex_t a = PI2 * I / m;
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

    const float* re = (float*)out;
    const float* im = (float*)out + 1;
    const size_t end_j = size >> 4;
    const __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    for (size_t m = 2; m <= size; m <<= 1)
    {
        const float a = PI2 / m;
        const size_t n = m >> 1;
        #pragma omp parallel for
        for (size_t j = 0; j < end_j; j++)
        {
            __m256i jj = _mm256_add_epi32(_mm256_set1_epi32(j << 3), ind);
            __m256i jn = _mm256_and_si256(jj, _mm256_set1_epi32(n - 1));
            __m256i i1 = _mm256_sub_epi32(_mm256_slli_epi32(jj, 1), jn);
            __m256i i2 = _mm256_add_epi32(i1, _mm256_set1_epi32(n));

            __m256 re1 = _mm256_i32gather_ps(re, i1, 8);
            __m256 re2 = _mm256_i32gather_ps(re, i2, 8);
            __m256 im1 = _mm256_i32gather_ps(im, i1, 8);
            __m256 im2 = _mm256_i32gather_ps(im, i2, 8);

            __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(jn), _mm256_set1_ps(a));
            __m256 s, c;
            sincos256_ps(w, &s, &c);

            __m256 tre = _mm256_sub_ps(_mm256_mul_ps(re2, c), _mm256_mul_ps(im2, s));
            __m256 tim = _mm256_add_ps(_mm256_mul_ps(re2, s), _mm256_mul_ps(im2, c));

            re2 = _mm256_sub_ps(re1, tre);
            re1 = _mm256_add_ps(re1, tre);
            im2 = _mm256_sub_ps(im1, tim);
            im1 = _mm256_add_ps(im1, tim);

            #define C(n,ii) out[_mm256_extract_epi32((i##ii),(n))]
            #define F(n,i) C(n, i) = CMPLXF((union vec8){re##i}.v[n], (union vec8){im##i}.v[n]);
            #define S(n) F(n,1); F(n,2);

            S(0); S(1); S(2); S(3); S(4); S(5); S(6); S(7);
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