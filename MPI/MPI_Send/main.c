#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <mpi.h>

typedef struct
{
    int size;
    int rank;
} comm_info_t;

comm_info_t init_mpi(int argc, char* argv[])
{
    comm_info_t mpi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    return mpi;
}

double get_time(comm_info_t mpi, int arr_size, int (*send_func)(void*, int, MPI_Datatype, int, int, MPI_Comm))
{
    int8_t* a = (int8_t*)malloc(arr_size);

    int s = arr_size + MPI_BSEND_OVERHEAD;
    int8_t* buf = (int8_t*)malloc(s);
    MPI_Buffer_attach(buf, s);

    double dt = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0)
    {
        double time0 = MPI_Wtime();
        send_func(a, arr_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        double time1 = MPI_Wtime();
        dt = time1 - time0;
    }
    else
    {
        sleep(2);
        MPI_Recv(a, arr_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Buffer_detach(&buf, &s);
    free(buf);

    free(a);

    return dt;
}

void test_send(int start, int end, int step, comm_info_t mpi, int (*send_func)(void*, int, MPI_Datatype, int, int, MPI_Comm))
{
    for (int size = start; size < end; size *= step)
    {
        double time = get_time(mpi, size, send_func);
        if (mpi.rank == 0) printf("%.6lf ", time);
    }
    if (mpi.rank == 0) printf("\n");
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    const int start = 10;
    const int end = 50000;
    const int step = 2;

    if (mpi.rank == 0)
    {
        printf("\t");
        for (int size = start; size < end; size *= step)
        {
            printf("%8d ", size);
        }
        printf("\n");
    }

    if (mpi.rank == 0) printf("MPI_Send  "); test_send(start, end, step, mpi, MPI_Send);
    if (mpi.rank == 0) printf("MPI_Ssend "); test_send(start, end, step, mpi, MPI_Ssend);
    if (mpi.rank == 0) printf("MPI_Rsend "); test_send(start, end, step, mpi, MPI_Rsend);
    if (mpi.rank == 0) printf("MPI_Bsend "); test_send(start, end, step, mpi, MPI_Bsend);

    for (int size = 250; size < 260; size += 1)
    {
        double time = get_time(mpi, size, MPI_Send);
        if (mpi.rank == 0) printf("MPI_Send %d %lf\n", size, time);
    }

    MPI_Finalize();
    return 0;
}