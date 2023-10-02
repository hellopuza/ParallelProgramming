#include "matrix.h"
#include <stdio.h>

typedef void(*matmul_t)(mat_t, mat_t, mat_t, size_t);

double test_mul(size_t size, matmul_t func)
{
    mat_t a, b, c;
    mat_create(a, size);
    mat_create(b, size);
    mat_create(c, size);
    mat_init(a, size);
    mat_init(b, size);

    double time = omp_get_wtime();
    func(a, b, c, size);
    time = omp_get_wtime() - time;

#ifdef DEBUG
    if (check_mul(a, b, c, size))
    {
        printf("Wrong multiplication\n");
        exit(1);
    }
#endif

    free(a);
    free(b);
    free(c);

    return time;
}

#define PRINT_MUL_TEST(func) \
    printf("%-30s : %lf\n", #func, test_mul(mat_size, func));

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Matrix size required\n");
        exit(1);
    }

    size_t mat_size = 0;
    sscanf(argv[1], "%lu", &mat_size);

    PRINT_MUL_TEST(mat_mul);
    PRINT_MUL_TEST(mat_mul_parallel);
    PRINT_MUL_TEST(mat_mul_opt_tran);
    PRINT_MUL_TEST(mat_mul_opt_tran_parallel);
    PRINT_MUL_TEST(mat_mul_opt_block);
    PRINT_MUL_TEST(mat_mul_opt_block_parallel);
    PRINT_MUL_TEST(mat_mul_strassen);
    PRINT_MUL_TEST(mat_mul_strassen_parallel);
    PRINT_MUL_TEST(mat_mul_strassen_simd);
    PRINT_MUL_TEST(mat_mul_strassen_simd_parallel);
}

