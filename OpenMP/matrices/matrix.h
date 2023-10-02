#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

int min(int a, int b)
{
    return a < b ? a : b;
}

typedef int* mat_t;

#define mat_create(mat, size) mat = (int*)calloc(sizeof(int*), size * size)

void mat_init(mat_t mat, size_t size)
{
    for (size_t i = 0; i < size * size; i++)
    {
        mat[i] = rand() % 10;
    }
}

#define at(mat, i, j) (mat)[(i) * (size) + (j)]

void mat_transpose(mat_t mat, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            int t = at(mat, i, j);
            at(mat, i, j) = at(mat, j, i);
            at(mat, j, i) = t;
        }
    }
}

void mat_transpose_parallel(mat_t mat, size_t size)
{
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            int t = at(mat, i, j);
            at(mat, i, j) = at(mat, j, i);
            at(mat, j, i) = t;
        }
    }
}

void mat_mul(mat_t a, mat_t b, mat_t res, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            at(res, i, j) = 0;
            for (size_t k = 0; k < size; k++)
            {
                at(res, i, j) += at(a, i, k) * at(b, k, j);
            }
        }
    }
}

void mat_mul_parallel(mat_t a, mat_t b, mat_t res, size_t size)
{
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            at(res, i, j) = 0;
            for (size_t k = 0; k < size; k++)
            {
                at(res, i, j) += at(a, i, k) * at(b, k, j);
            }
        }
    }
}

void mat_mul_opt_tran(mat_t a, mat_t b, mat_t res, size_t size)
{
    mat_transpose(b, size);
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            int sum = 0;
            for (size_t k = 0; k < size; k++)
            {
                sum += at(a, i, k) * at(b, j, k);
            }
            at(res, i, j) = sum;
        }
    }
    mat_transpose(b, size);
}

void mat_mul_opt_tran_parallel(mat_t a, mat_t b, mat_t res, size_t size)
{
    mat_transpose_parallel(b, size);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            int sum = 0;
            for (size_t k = 0; k < size; k++)
            {
                sum += at(a, i, k) * at(b, j, k);
            }
            at(res, i, j) = sum;
        }
    }
    mat_transpose_parallel(b, size);
}

const size_t bsize = 64;

void mat_mul_opt_block(mat_t a, mat_t b, mat_t res, size_t size)
{
    for (size_t bj = 0; bj < size; bj += bsize)
    {
        size_t maxj = min(bj + bsize, size);
        for (size_t bk = 0; bk < size; bk += bsize)
        {
            size_t maxk = min(bk + bsize, size);
            for (size_t i = 0; i < size; i++)
            {
                for (size_t j = bj; j < maxj; j++)
                {
                    int sum = 0;
                    for (size_t k = bk; k < maxk; k++)
                    {
                        sum += at(a, i, k) * at(b, k, j);
                    }
                    at(res, i, j) += sum;
                }
            }
        }
    }
}

void mat_mul_opt_block_parallel(mat_t a, mat_t b, mat_t res, size_t size)
{
    #pragma omp parallel for collapse(2)
    for (size_t bj = 0; bj < size; bj += bsize)
    {
        for (size_t bk = 0; bk < size; bk += bsize)
        {
            size_t maxj = min(bj + bsize, size);
            size_t maxk = min(bk + bsize, size);
            for (size_t i = 0; i < size; i++)
            {
                for (size_t j = bj; j < maxj; j++)
                {
                    int sum = 0;
                    for (size_t k = bk; k < maxk; k++)
                    {
                        sum += at(a, i, k) * at(b, k, j);
                    }
                    at(res, i, j) += sum;
                }
            }
        }
    }
}

void mat_split2x2(mat_t a, size_t size, mat_t s[2][2], int* data)
{
    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;

    s[0][0] = data;
    s[0][1] = s[0][0] + msize;
    s[1][0] = s[0][1] + msize;
    s[1][1] = s[1][0] + msize;

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            s[i / hsize][j / hsize][(i % hsize) * hsize + j % hsize] = a[i * size + j];
        }
    }
}

void mat_split2x2_parallel(mat_t a, size_t size, mat_t s[2][2], int* data)
{
    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;

    s[0][0] = data;
    s[0][1] = s[0][0] + msize;
    s[1][0] = s[0][1] + msize;
    s[1][1] = s[1][0] + msize;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            s[i / hsize][j / hsize][(i % hsize) * hsize + j % hsize] = a[i * size + j];
        }
    }
}

void mat_mul_strassen(mat_t a, mat_t b, mat_t res, size_t size)
{
    if (size <= 256)
    {
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                int sum = 0;
                for (size_t k = 0; k < size; k++)
                {
                    sum += at(a, i, k) * at(b, k, j);
                }
                at(res, i, j) = sum;
            }
        }
        return;
    }

    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;
    int* data = (int*)calloc(sizeof(int), msize * 25);

    mat_t as[2][2];
    mat_t bs[2][2];
    mat_split2x2(a, size, as, data);
    mat_split2x2(b, size, bs, data + 4 * msize);

    mat_t am[5];
    mat_t bm[5];
    for (int i = 0; i < 5; i++)
    {
        am[i] = data + (8 + i) * msize;
        bm[i] = data + (13 + i) * msize;
    }

    mat_t m[7];
    for (int i = 0; i < 7; i++)
    {
        m[i] = data + (18 + i) * msize;
    }

    for (size_t ind = 0; ind < msize; ind++)
    {
        am[0][ind] = as[0][0][ind] + as[1][1][ind];
        am[1][ind] = as[1][0][ind] + as[1][1][ind];
        am[2][ind] = as[0][0][ind] + as[0][1][ind];
        am[3][ind] = as[1][0][ind] - as[0][0][ind];
        am[4][ind] = as[0][1][ind] - as[1][1][ind];

        bm[0][ind] = bs[0][0][ind] + bs[1][1][ind];
        bm[1][ind] = bs[0][1][ind] - bs[1][1][ind];
        bm[2][ind] = bs[1][0][ind] - bs[0][0][ind];
        bm[3][ind] = bs[0][0][ind] + bs[0][1][ind];
        bm[4][ind] = bs[1][0][ind] + bs[1][1][ind];
    }

    mat_mul_strassen(am[0], bm[0], m[0], hsize);
    mat_mul_strassen(am[1], bs[0][0], m[1], hsize);
    mat_mul_strassen(as[0][0], bm[1], m[2], hsize);
    mat_mul_strassen(as[1][1], bm[2], m[3], hsize);
    mat_mul_strassen(am[2], bs[1][1], m[4], hsize);
    mat_mul_strassen(am[3], bm[3], m[5], hsize);
    mat_mul_strassen(am[4], bm[4], m[6], hsize);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            size_t ind = (i % hsize) * hsize + j % hsize;
            size_t n = (i / hsize) * 2 + j / hsize;

            switch (n)
            {
            case 0: at(res, i, j) = m[0][ind] + m[3][ind] - m[4][ind] + m[6][ind]; break;
            case 1: at(res, i, j) = m[2][ind] + m[4][ind]; break;
            case 2: at(res, i, j) = m[1][ind] + m[3][ind]; break;
            case 3: at(res, i, j) = m[0][ind] - m[1][ind] + m[2][ind] + m[5][ind]; break;
            }
        }
    }

    free(data);
}

void mat_mul_strassen_parallel(mat_t a, mat_t b, mat_t res, size_t size)
{
    if (size <= 256)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                int sum = 0;
                for (size_t k = 0; k < size; k++)
                {
                    sum += at(a, i, k) * at(b, k, j);
                }
                at(res, i, j) = sum;
            }
        }
        return;
    }

    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;
    int* data = (int*)calloc(sizeof(int), msize * 25);

    mat_t as[2][2];
    mat_t bs[2][2];
    mat_split2x2_parallel(a, size, as, data);
    mat_split2x2_parallel(b, size, bs, data + 4 * msize);

    mat_t am[5];
    mat_t bm[5];
    for (int i = 0; i < 5; i++)
    {
        am[i] = data + (8 + i) * msize;
        bm[i] = data + (13 + i) * msize;
    }

    mat_t m[7];
    for (int i = 0; i < 7; i++)
    {
        m[i] = data + (18 + i) * msize;
    }

    #pragma omp parallel for
    for (size_t ind = 0; ind < msize; ind++)
    {
        am[0][ind] = as[0][0][ind] + as[1][1][ind];
        am[1][ind] = as[1][0][ind] + as[1][1][ind];
        am[2][ind] = as[0][0][ind] + as[0][1][ind];
        am[3][ind] = as[1][0][ind] - as[0][0][ind];
        am[4][ind] = as[0][1][ind] - as[1][1][ind];

        bm[0][ind] = bs[0][0][ind] + bs[1][1][ind];
        bm[1][ind] = bs[0][1][ind] - bs[1][1][ind];
        bm[2][ind] = bs[1][0][ind] - bs[0][0][ind];
        bm[3][ind] = bs[0][0][ind] + bs[0][1][ind];
        bm[4][ind] = bs[1][0][ind] + bs[1][1][ind];
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        mat_mul_strassen(am[0], bm[0], m[0], hsize);
        #pragma omp section
        mat_mul_strassen(am[1], bs[0][0], m[1], hsize);
        #pragma omp section
        mat_mul_strassen(as[0][0], bm[1], m[2], hsize);
        #pragma omp section
        mat_mul_strassen(as[1][1], bm[2], m[3], hsize);
        #pragma omp section
        mat_mul_strassen(am[2], bs[1][1], m[4], hsize);
        #pragma omp section
        mat_mul_strassen(am[3], bm[3], m[5], hsize);
        #pragma omp section
        mat_mul_strassen(am[4], bm[4], m[6], hsize);
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            size_t ind = (i % hsize) * hsize + j % hsize;
            size_t n = (i / hsize) * 2 + j / hsize;

            switch (n)
            {
            case 0: at(res, i, j) = m[0][ind] + m[3][ind] - m[4][ind] + m[6][ind]; break;
            case 1: at(res, i, j) = m[2][ind] + m[4][ind]; break;
            case 2: at(res, i, j) = m[1][ind] + m[3][ind]; break;
            case 3: at(res, i, j) = m[0][ind] - m[1][ind] + m[2][ind] + m[5][ind]; break;
            }
        }
    }

    free(data);
}

void mat_mul_strassen_simd(mat_t a, mat_t b, mat_t res, size_t size)
{
    if (size <= 256)
    {
        mat_transpose(b, size);
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                __m256i sumi = _mm256_setzero_si256();
                size_t k = 0;
                for (; k + 8 <= size; k += 8)
                {
                    __m256i ai = _mm256_loadu_si256((__m256i*)&(at(a, i, k)));
                    __m256i bi = _mm256_loadu_si256((__m256i*)&(at(b, j, k)));
                    sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(ai, bi));
                }
                int sum = 0;
                for (k; k < size; k++)
                {
                    sum += at(a, i, k) * at(b, j, k);
                }
                for (int i = 0; i < 8; i++)
                {
                    sum += ((int*)&sumi)[i];
                }
                at(res, i, j) = sum;
            }
        }
        mat_transpose(b, size);
        return;
    }

    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;
    int* data = (int*)calloc(sizeof(int), msize * 25);

    mat_t as[2][2];
    mat_t bs[2][2];
    mat_split2x2(a, size, as, data);
    mat_split2x2(b, size, bs, data + 4 * msize);

    mat_t am[5];
    mat_t bm[5];
    for (int i = 0; i < 5; i++)
    {
        am[i] = data + (8 + i) * msize;
        bm[i] = data + (13 + i) * msize;
    }

    mat_t m[7];
    for (int i = 0; i < 7; i++)
    {
        m[i] = data + (18 + i) * msize;
    }

    size_t ind = 0;
    for (; ind + 8 <= msize; ind += 8)
    {
        __m256i as00 = _mm256_loadu_si256((__m256i*)&(as[0][0][ind]));
        __m256i as01 = _mm256_loadu_si256((__m256i*)&(as[0][1][ind]));
        __m256i as10 = _mm256_loadu_si256((__m256i*)&(as[1][0][ind]));
        __m256i as11 = _mm256_loadu_si256((__m256i*)&(as[1][1][ind]));
        __m256i bs00 = _mm256_loadu_si256((__m256i*)&(bs[0][0][ind]));
        __m256i bs01 = _mm256_loadu_si256((__m256i*)&(bs[0][1][ind]));
        __m256i bs10 = _mm256_loadu_si256((__m256i*)&(bs[1][0][ind]));
        __m256i bs11 = _mm256_loadu_si256((__m256i*)&(bs[1][1][ind]));

        __m256i am0 = _mm256_add_epi32(as00, as11);
        __m256i am1 = _mm256_add_epi32(as10, as11);
        __m256i am2 = _mm256_add_epi32(as00, as01);
        __m256i am3 = _mm256_sub_epi32(as10, as00);
        __m256i am4 = _mm256_sub_epi32(as01, as11);

        __m256i bm0 = _mm256_add_epi32(bs00, bs11);
        __m256i bm1 = _mm256_sub_epi32(bs01, bs11);
        __m256i bm2 = _mm256_sub_epi32(bs10, bs00);
        __m256i bm3 = _mm256_add_epi32(bs00, bs01);
        __m256i bm4 = _mm256_add_epi32(bs10, bs11);

        _mm256_storeu_si256((__m256i*)&(am[0][ind]), am0);
        _mm256_storeu_si256((__m256i*)&(am[1][ind]), am1);
        _mm256_storeu_si256((__m256i*)&(am[2][ind]), am2);
        _mm256_storeu_si256((__m256i*)&(am[3][ind]), am3);
        _mm256_storeu_si256((__m256i*)&(am[4][ind]), am4);
        _mm256_storeu_si256((__m256i*)&(bm[0][ind]), bm0);
        _mm256_storeu_si256((__m256i*)&(bm[1][ind]), bm1);
        _mm256_storeu_si256((__m256i*)&(bm[2][ind]), bm2);
        _mm256_storeu_si256((__m256i*)&(bm[3][ind]), bm3);
        _mm256_storeu_si256((__m256i*)&(bm[4][ind]), bm4);
    }
    for (; ind < msize; ind++)
    {
        am[0][ind] = as[0][0][ind] + as[1][1][ind];
        am[1][ind] = as[1][0][ind] + as[1][1][ind];
        am[2][ind] = as[0][0][ind] + as[0][1][ind];
        am[3][ind] = as[1][0][ind] - as[0][0][ind];
        am[4][ind] = as[0][1][ind] - as[1][1][ind];

        bm[0][ind] = bs[0][0][ind] + bs[1][1][ind];
        bm[1][ind] = bs[0][1][ind] - bs[1][1][ind];
        bm[2][ind] = bs[1][0][ind] - bs[0][0][ind];
        bm[3][ind] = bs[0][0][ind] + bs[0][1][ind];
        bm[4][ind] = bs[1][0][ind] + bs[1][1][ind];
    }

    mat_mul_strassen_simd(am[0], bm[0], m[0], hsize);
    mat_mul_strassen_simd(am[1], bs[0][0], m[1], hsize);
    mat_mul_strassen_simd(as[0][0], bm[1], m[2], hsize);
    mat_mul_strassen_simd(as[1][1], bm[2], m[3], hsize);
    mat_mul_strassen_simd(am[2], bs[1][1], m[4], hsize);
    mat_mul_strassen_simd(am[3], bm[3], m[5], hsize);
    mat_mul_strassen_simd(am[4], bm[4], m[6], hsize);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            size_t ind = (i % hsize) * hsize + j % hsize;
            size_t n = (i / hsize) * 2 + j / hsize;

            switch (n)
            {
            case 0: at(res, i, j) = m[0][ind] + m[3][ind] - m[4][ind] + m[6][ind]; break;
            case 1: at(res, i, j) = m[2][ind] + m[4][ind]; break;
            case 2: at(res, i, j) = m[1][ind] + m[3][ind]; break;
            case 3: at(res, i, j) = m[0][ind] - m[1][ind] + m[2][ind] + m[5][ind]; break;
            }
        }
    }

    free(data);
}

void mat_mul_strassen_simd_parallel(mat_t a, mat_t b, mat_t res, size_t size)
{
    if (size <= 256)
    {
        mat_transpose_parallel(b, size);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                __m256i sumi = _mm256_setzero_si256();
                size_t k = 0;
                for (; k + 8 <= size; k += 8)
                {
                    __m256i ai = _mm256_loadu_si256((__m256i*)&(at(a, i, k)));
                    __m256i bi = _mm256_loadu_si256((__m256i*)&(at(b, j, k)));
                    sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(ai, bi));
                }
                int sum = 0;
                for (k; k < size; k++)
                {
                    sum += at(a, i, k) * at(b, j, k);
                }
                for (int i = 0; i < 8; i++)
                {
                    sum += ((int*)&sumi)[i];
                }
                at(res, i, j) = sum;
            }
        }
        mat_transpose(b, size);
        return;
    }

    size_t hsize = size / 2 + size % 2;
    size_t msize = hsize * hsize;
    int* data = (int*)calloc(sizeof(int), msize * 25);

    mat_t as[2][2];
    mat_t bs[2][2];
    mat_split2x2_parallel(a, size, as, data);
    mat_split2x2_parallel(b, size, bs, data + 4 * msize);

    mat_t am[5];
    mat_t bm[5];
    for (int i = 0; i < 5; i++)
    {
        am[i] = data + (8 + i) * msize;
        bm[i] = data + (13 + i) * msize;
    }

    mat_t m[7];
    for (int i = 0; i < 7; i++)
    {
        m[i] = data + (18 + i) * msize;
    }

    size_t last_ind = msize - 8;
    #pragma omp parallel for
    for (size_t ind = 0; ind <= last_ind; ind += 8)
    {
        __m256i as00 = _mm256_loadu_si256((__m256i*)&(as[0][0][ind]));
        __m256i as01 = _mm256_loadu_si256((__m256i*)&(as[0][1][ind]));
        __m256i as10 = _mm256_loadu_si256((__m256i*)&(as[1][0][ind]));
        __m256i as11 = _mm256_loadu_si256((__m256i*)&(as[1][1][ind]));
        __m256i bs00 = _mm256_loadu_si256((__m256i*)&(bs[0][0][ind]));
        __m256i bs01 = _mm256_loadu_si256((__m256i*)&(bs[0][1][ind]));
        __m256i bs10 = _mm256_loadu_si256((__m256i*)&(bs[1][0][ind]));
        __m256i bs11 = _mm256_loadu_si256((__m256i*)&(bs[1][1][ind]));

        __m256i am0 = _mm256_add_epi32(as00, as11);
        __m256i am1 = _mm256_add_epi32(as10, as11);
        __m256i am2 = _mm256_add_epi32(as00, as01);
        __m256i am3 = _mm256_sub_epi32(as10, as00);
        __m256i am4 = _mm256_sub_epi32(as01, as11);

        __m256i bm0 = _mm256_add_epi32(bs00, bs11);
        __m256i bm1 = _mm256_sub_epi32(bs01, bs11);
        __m256i bm2 = _mm256_sub_epi32(bs10, bs00);
        __m256i bm3 = _mm256_add_epi32(bs00, bs01);
        __m256i bm4 = _mm256_add_epi32(bs10, bs11);

        _mm256_storeu_si256((__m256i*)&(am[0][ind]), am0);
        _mm256_storeu_si256((__m256i*)&(am[1][ind]), am1);
        _mm256_storeu_si256((__m256i*)&(am[2][ind]), am2);
        _mm256_storeu_si256((__m256i*)&(am[3][ind]), am3);
        _mm256_storeu_si256((__m256i*)&(am[4][ind]), am4);
        _mm256_storeu_si256((__m256i*)&(bm[0][ind]), bm0);
        _mm256_storeu_si256((__m256i*)&(bm[1][ind]), bm1);
        _mm256_storeu_si256((__m256i*)&(bm[2][ind]), bm2);
        _mm256_storeu_si256((__m256i*)&(bm[3][ind]), bm3);
        _mm256_storeu_si256((__m256i*)&(bm[4][ind]), bm4);
    }
    for (size_t ind = (msize / 8) * 8; ind < msize; ind++)
    {
        am[0][ind] = as[0][0][ind] + as[1][1][ind];
        am[1][ind] = as[1][0][ind] + as[1][1][ind];
        am[2][ind] = as[0][0][ind] + as[0][1][ind];
        am[3][ind] = as[1][0][ind] - as[0][0][ind];
        am[4][ind] = as[0][1][ind] - as[1][1][ind];

        bm[0][ind] = bs[0][0][ind] + bs[1][1][ind];
        bm[1][ind] = bs[0][1][ind] - bs[1][1][ind];
        bm[2][ind] = bs[1][0][ind] - bs[0][0][ind];
        bm[3][ind] = bs[0][0][ind] + bs[0][1][ind];
        bm[4][ind] = bs[1][0][ind] + bs[1][1][ind];
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        mat_mul_strassen_simd(am[0], bm[0], m[0], hsize);
        #pragma omp section
        mat_mul_strassen_simd(am[1], bs[0][0], m[1], hsize);
        #pragma omp section
        mat_mul_strassen_simd(as[0][0], bm[1], m[2], hsize);
        #pragma omp section
        mat_mul_strassen_simd(as[1][1], bm[2], m[3], hsize);
        #pragma omp section
        mat_mul_strassen_simd(am[2], bs[1][1], m[4], hsize);
        #pragma omp section
        mat_mul_strassen_simd(am[3], bm[3], m[5], hsize);
        #pragma omp section
        mat_mul_strassen_simd(am[4], bm[4], m[6], hsize);
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            size_t ind = (i % hsize) * hsize + j % hsize;
            size_t n = (i / hsize) * 2 + j / hsize;

            switch (n)
            {
            case 0: at(res, i, j) = m[0][ind] + m[3][ind] - m[4][ind] + m[6][ind]; break;
            case 1: at(res, i, j) = m[2][ind] + m[4][ind]; break;
            case 2: at(res, i, j) = m[1][ind] + m[3][ind]; break;
            case 3: at(res, i, j) = m[0][ind] - m[1][ind] + m[2][ind] + m[5][ind]; break;
            }
        }
    }

    free(data);
}


int check_mul(mat_t a, mat_t b, mat_t res, size_t size)
{
    mat_t c;
    mat_create(c, size);
    mat_mul(a, b, c, size);

    for (size_t i = 0; i < size * size; i++)
    {
        if (res[i] != c[i])
        {
            return 1;
        }
    }

    free(c);
    return 0;
}

