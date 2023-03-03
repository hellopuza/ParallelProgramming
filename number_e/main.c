#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>
#include <gmp.h>
#include <mpfr.h>

typedef struct
{
    int size;
    int rank;
} comm_info_t;

typedef struct
{
    mpz_t p;
    mpz_t q;
} bsr_term_t;

comm_info_t init_mpi(int argc, char* argv[])
{
    comm_info_t mpi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    return mpi;
}

void send_mpz(mpz_t num, int dest, int tag, MPI_Comm comm)
{
    uint32_t size = (mpz_sizeinbase(num, 2) - 1) / 8 + 1;
    uint32_t count = size / sizeof(uint32_t);
    count = (count == 0) ? 1 : count;
    uint32_t* ptr = (uint32_t*)malloc(size);

    mpz_export(ptr, NULL, 1, sizeof(uint32_t), 0, 0, num);
    MPI_Send(&size, 1, MPI_UNSIGNED, dest, tag, comm);
    MPI_Send(ptr, count, MPI_UNSIGNED, dest, tag, comm);
    free(ptr);
}

void recv_mpz(mpz_t num, int source, int tag, MPI_Comm comm)
{
    uint32_t size = 0;
    MPI_Recv(&size, 1, MPI_UNSIGNED, source, tag, comm, MPI_STATUS_IGNORE);
    uint32_t count = size / sizeof(uint32_t);
    count = (count == 0) ? 1 : count;

    uint32_t* ptr = (uint32_t*)malloc(size);
    MPI_Recv(ptr, count, MPI_UNSIGNED, source, tag, comm, MPI_STATUS_IGNORE);
    mpz_import(num, count, 1, sizeof(uint32_t), 0, 0, ptr);
    free(ptr);
}

uint32_t bits_estimation(uint32_t num)
{
    float n = (float)num;
    return (uint32_t)ceilf((3.5F + n) * log2f(n) + log2f(sqrtf(2.0F * M_PI)) - n * log2f(M_E));
}

uint32_t error_estimation(uint32_t x)
{
    float xf = (float)x;
    return (uint32_t)floorf((1.5F + xf) * log10f(xf) + log10f(sqrtf(2.0F * M_PI)) - xf * log10f(M_E));
}

uint32_t get_number_of_sum_elements(uint32_t digits_num)
{
    digits_num++;
    uint32_t a = 1;
    uint32_t b = (digits_num > 21) ? digits_num : 21;
    while (b - a > 1)
    {
        uint32_t mid = (a + b) / 2;
        (error_estimation(mid) < digits_num) ? (a = mid) : (b = mid);
    }
    return a;
}

void calc_e_parallel(mpfr_t e, uint32_t num, comm_info_t mpi)
{
    uint32_t range = num / mpi.size;
    uint32_t first_term = mpi.rank * range + 1;
    uint32_t last_term = (mpi.rank == mpi.size - 1) ? num : (mpi.rank + 1) * range;
    uint32_t terms_num = last_term - first_term + 1;
    bsr_term_t* terms = (bsr_term_t*)malloc(sizeof(bsr_term_t) * terms_num);
    for (uint32_t i = first_term; i <= last_term; i++)
    {
        mpz_init_set_ui(terms[i - first_term].p, 1);
        mpz_init_set_ui(terms[i - first_term].q, i);
    }

    while (terms_num != 1)
    {
        for (uint32_t i = 0; i < terms_num - 1; i += 2)
        {
            mpz_mul(terms[i].p, terms[i].p, terms[i + 1].q);
            mpz_add(terms[i / 2].p, terms[i].p, terms[i + 1].p);
            mpz_mul(terms[i / 2].q, terms[i].q, terms[i + 1].q);
        }
        if (terms_num % 2 == 1)
        {
            mpz_set(terms[terms_num / 2].p, terms[terms_num - 1].p);
            mpz_set(terms[terms_num / 2].q, terms[terms_num - 1].q);
        }

        for (uint32_t i = (terms_num / 2) + (terms_num % 2); i < terms_num; i++)
        {
            mpz_clears(terms[i].p, terms[i].q, NULL);
        }

        terms_num = (terms_num / 2) + (terms_num % 2);
    }

    if (mpi.rank == 0)
    {
        mpz_t p, q;
        mpz_inits(p, q, NULL);
        for (int i = 1; i < mpi.size; i++)
        {
            recv_mpz(p, i, 0, MPI_COMM_WORLD);
            recv_mpz(q, i, 0, MPI_COMM_WORLD);

            mpz_mul(terms[0].p, terms[0].p, q);
            mpz_add(terms[0].p, terms[0].p, p);
            mpz_mul(terms[0].q, terms[0].q, q);
        }
        mpz_clears(p, q, NULL);

        mpfr_set_z(e, terms[0].p, MPFR_RNDZ);
        mpfr_div_z(e, e, terms[0].q, MPFR_RNDZ);
        mpfr_add_ui(e, e, 1, MPFR_RNDZ);
    }
    else
    {
        send_mpz(terms[0].p, 0, 0, MPI_COMM_WORLD);
        send_mpz(terms[0].q, 0, 0, MPI_COMM_WORLD);
    }

    mpz_clears(terms[0].p, terms[0].q, NULL);
    free(terms);
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    uint32_t digits = 0;
    sscanf(argv[1], "%u", &digits);
    uint32_t num = get_number_of_sum_elements(digits);
    mpfr_set_default_prec(bits_estimation(num));

    mpfr_t e;
    mpfr_init(e);

    double time0 = MPI_Wtime();
    calc_e_parallel(e, num, mpi);
    double time1 = MPI_Wtime();
    //if (mpi.rank == 0) mpfr_printf("%.5lf %.*Rf\n", time1 - time0, (int)digits, e);
    printf("%d %.8lf\n", mpi.rank, time1 - time0);

    mpfr_clear(e);

    MPI_Finalize();
    return 0;
}