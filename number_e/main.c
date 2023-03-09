#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>
#include <gmp.h>
#include <mpfr.h>

#define M_PI 3.14159265358979323846
#define M_E  2.71828182845904523536

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
    MPI_Send(&num->_mp_alloc, 1, MPI_INT, dest, tag, comm);
    MPI_Send(&num->_mp_size, 1, MPI_INT, dest, tag, comm);
    MPI_Send(num->_mp_d, num->_mp_alloc * sizeof(mp_limb_t), MPI_BYTE, dest, tag, comm);
}

void recv_mpz(mpz_t num, int source, int tag, MPI_Comm comm)
{
    MPI_Recv(&num->_mp_alloc, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&num->_mp_size, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);

    num->_mp_d = (mp_limb_t*)calloc(num->_mp_alloc, sizeof(mp_limb_t));
    MPI_Recv(num->_mp_d, num->_mp_alloc * sizeof(mp_limb_t), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
}

uint32_t bits_estimation(uint32_t d)
{
    return 64 + (uint32_t)ceilf(log2f(10.0F) * d);
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

void calc_e(mpfr_t e, uint32_t num, comm_info_t mpi)
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

    mpz_t p, q;
    int step = 1;
    while (step < mpi.size)
    {
        if ((mpi.rank % (step * 2) == 0) && (mpi.rank < mpi.size - step))
        {
            recv_mpz(p, mpi.rank + step, 0, MPI_COMM_WORLD);
            recv_mpz(q, mpi.rank + step, 0, MPI_COMM_WORLD);

            mpz_mul(terms[0].p, terms[0].p, q);
            mpz_add(terms[0].p, terms[0].p, p);
            mpz_mul(terms[0].q, terms[0].q, q);

            free(p->_mp_d);
            free(q->_mp_d);
        }
        else if (mpi.rank % (step * 2) == step)
        {
            send_mpz(terms[0].p, mpi.rank - step, 0, MPI_COMM_WORLD);
            send_mpz(terms[0].q, mpi.rank - step, 0, MPI_COMM_WORLD);
        }
        step *= 2;
    }
    if (mpi.rank == 0)
    {
        mpfr_set_z(e, terms[0].p, MPFR_RNDZ);
        mpfr_div_z(e, e, terms[0].q, MPFR_RNDZ);
        mpfr_add_ui(e, e, 1, MPFR_RNDZ);
    }

    mpz_clears(terms[0].p, terms[0].q, NULL);
    free(terms);
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);
    if (argc < 2)
    {
        printf("Input required\n");
        MPI_Finalize();
        return 1;
    }

    uint32_t digits = 0;
    sscanf(argv[1], "%u", &digits);
    uint32_t num = get_number_of_sum_elements(digits);
    mpfr_set_default_prec(bits_estimation(digits));

    mpfr_t e;
    mpfr_init(e);

    double time0 = MPI_Wtime();
    calc_e(e, num, mpi);
    double time1 = MPI_Wtime();

#ifdef CALC_TIME
    if (mpi.rank == 0) printf("%.6lf\n", time1 - time0);
#else
    if (mpi.rank == 0) mpfr_printf("%.*Rf\n", (int)digits, e);
#endif

    mpfr_clear(e);

    MPI_Finalize();
    return 0;
}